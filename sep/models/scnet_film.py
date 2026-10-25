"""Query-conditioned SCNet (Sparse Compression Network, Tong et al., ICASSP 2024,
https://arxiv.org/abs/2401.13276).

Building blocks are copied from the official implementation
https://github.com/starrytong/SCNet (MIT License, Copyright (c) 2024 starrytong),
with three changes for this project:
  1. one output stem instead of a fixed list of sources;
  2. FiLM conditioning on the requested instrument after every dual-path layer
     (identity at initialisation);
  3. STFT / iSTFT and the input normalisation run in fp32 under autocast.

SCNet predicts the complex target spectrogram directly.
"""
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------- official SCNet blocks
class Swish(nn.Module):
    def forward(self, x):
        return x * x.sigmoid()


class ConvolutionModule(nn.Module):
    def __init__(self, channels, depth=2, compress=4, kernel=3):
        super().__init__()
        assert kernel % 2 == 1
        hidden = int(channels / compress)
        norm = lambda d: nn.GroupNorm(1, d)  # noqa: E731
        self.layers = nn.ModuleList([
            nn.Sequential(
                norm(channels),
                nn.Conv1d(channels, hidden * 2, kernel, padding=kernel // 2),
                nn.GLU(1),
                nn.Conv1d(hidden, hidden, kernel, padding=kernel // 2, groups=hidden),
                norm(hidden),
                Swish(),
                nn.Conv1d(hidden, channels, 1),
            ) for _ in range(abs(depth))])

    def forward(self, x):
        for layer in self.layers:
            x = x + layer(x)
        return x


class FusionLayer(nn.Module):
    def __init__(self, channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(channels * 2, channels * 2, kernel_size, stride=stride, padding=padding)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + skip
        x = x.repeat(1, 2, 1, 1)
        return F.glu(self.conv(x), dim=1)


class SDlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()
        self.convs = nn.ModuleList()
        self.strides, self.kernels = [], []
        for cfg in band_configs.values():
            self.convs.append(nn.Conv2d(channels_in, channels_out, (cfg["kernel"], 1), (cfg["stride"], 1), (0, 0)))
            self.strides.append(cfg["stride"])
            self.kernels.append(cfg["kernel"])
        self.SR_low = band_configs["low"]["SR"]
        self.SR_mid = band_configs["mid"]["SR"]

    def forward(self, x):
        Fr = x.shape[2]
        splits = [(0, math.ceil(Fr * self.SR_low)),
                  (math.ceil(Fr * self.SR_low), math.ceil(Fr * (self.SR_low + self.SR_mid))),
                  (math.ceil(Fr * (self.SR_low + self.SR_mid)), Fr)]
        outputs, original_lengths = [], []
        for conv, stride, kernel, (start, end) in zip(self.convs, self.strides, self.kernels, splits):
            band = x[:, :, start:end, :]
            original_lengths.append(end - start)
            n = band.shape[2]
            total = kernel - stride if stride == 1 else (stride - n % stride) % stride
            band = F.pad(band, (0, 0, total // 2, total - total // 2))
            outputs.append(conv(band))
        return outputs, original_lengths


class SUlayer(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs):
        super().__init__()
        self.convtrs = nn.ModuleList([
            nn.ConvTranspose2d(channels_in, channels_out, [cfg["kernel"], 1], [cfg["stride"], 1])
            for cfg in band_configs.values()])

    def forward(self, x, lengths, origin_lengths):
        splits = [(0, lengths[0]), (lengths[0], lengths[0] + lengths[1]), (lengths[0] + lengths[1], None)]
        outputs = []
        for idx, (convtr, (start, end)) in enumerate(zip(self.convtrs, splits)):
            out = convtr(x[:, :, start:end, :])
            dist = abs(origin_lengths[idx] - out.shape[2]) // 2
            outputs.append(out[:, :, dist:dist + origin_lengths[idx], :])
        return torch.cat(outputs, dim=2)


class SDblock(nn.Module):
    def __init__(self, channels_in, channels_out, band_configs, conv_config, depths=(3, 2, 1), kernel_size=3):
        super().__init__()
        self.SDlayer = SDlayer(channels_in, channels_out, band_configs)
        self.conv_modules = nn.ModuleList([ConvolutionModule(channels_out, d, **conv_config) for d in depths])
        self.globalconv = nn.Conv2d(channels_out, channels_out, kernel_size, 1, (kernel_size - 1) // 2)

    def forward(self, x):
        bands, original_lengths = self.SDlayer(x)
        bands = [
            F.gelu(conv(b.permute(0, 2, 1, 3).reshape(-1, b.shape[1], b.shape[3]))
                   .view(b.shape[0], b.shape[2], b.shape[1], b.shape[3]).permute(0, 2, 1, 3))
            for conv, b in zip(self.conv_modules, bands)]
        lengths = [b.size(-2) for b in bands]
        full = torch.cat(bands, dim=2)
        return self.globalconv(full), full, lengths, original_lengths


class FeatureConversion(nn.Module):
    def __init__(self, channels, inverse):
        super().__init__()
        self.inverse, self.channels = inverse, channels

    def forward(self, x):
        x = x.float()
        if self.inverse:
            x = torch.complex(x[:, :self.channels // 2], x[:, self.channels // 2:])
            return torch.fft.irfft(x, dim=3, norm="ortho")
        x = torch.fft.rfft(x, dim=3, norm="ortho")
        return torch.cat([x.real, x.imag], dim=1)


class DualPathRNN(nn.Module):
    def __init__(self, d_model, expand, bidirectional=True):
        super().__init__()
        h = d_model * expand
        self.lstm_layers = nn.ModuleList([nn.LSTM(d_model, h, batch_first=True, bidirectional=bidirectional)
                                          for _ in range(2)])
        self.linear_layers = nn.ModuleList([nn.Linear(h * 2, d_model) for _ in range(2)])
        self.norm_layers = nn.ModuleList([nn.GroupNorm(1, d_model) for _ in range(2)])

    def forward(self, x):
        B, C, Fr, T = x.shape
        res = x                                                    # frequency path
        x = self.norm_layers[0](x).transpose(1, 3).contiguous().view(B * T, Fr, C)
        x = self.linear_layers[0](self.lstm_layers[0](x)[0])
        x = x.view(B, T, Fr, C).transpose(1, 3) + res
        res = x                                                    # time path
        x = self.norm_layers[1](x).transpose(1, 2).contiguous().view(B * Fr, C, T).transpose(1, 2)
        x = self.linear_layers[1](self.lstm_layers[1](x)[0])
        return x.transpose(1, 2).contiguous().view(B, Fr, C, T).transpose(1, 2) + res


# ----------------------------------------------------------------- conditioned model
class SCNetFiLM(nn.Module):
    def __init__(self, n_queries=4, audio_channels=2, dims=(4, 32, 64, 128), nfft=4096, hop_size=1024,
                 win_size=4096, normalized=True, band_SR=(0.175, 0.392, 0.433), band_stride=(1, 4, 16),
                 band_kernel=(3, 4, 16), conv_depths=(3, 2, 1), compress=4, conv_kernel=3,
                 num_dplayer=6, expand=1, cond_dim=64):
        super().__init__()
        assert dims[0] == 2 * audio_channels
        self.audio_channels, self.dims, self.hop_length = audio_channels, list(dims), hop_size
        keys = ["low", "mid", "high"]
        bc = {keys[i]: {"SR": band_SR[i], "stride": band_stride[i], "kernel": band_kernel[i]} for i in range(3)}
        cc = {"compress": compress, "kernel": conv_kernel}
        self.stft_config = dict(n_fft=nfft, hop_length=hop_size, win_length=win_size, center=True,
                                normalized=normalized)
        self.register_buffer("window", torch.hann_window(win_size), persistent=False)

        self.encoder, self.decoder = nn.ModuleList(), nn.ModuleList()
        for i in range(len(dims) - 1):
            self.encoder.append(SDblock(dims[i], dims[i + 1], bc, cc, conv_depths))
            self.decoder.insert(0, nn.Sequential(FusionLayer(dims[i + 1]), SUlayer(dims[i + 1], dims[i], bc)))

        C = dims[-1]
        self.dp = nn.ModuleList([DualPathRNN(C * (2 if i % 2 else 1), expand) for i in range(num_dplayer)])
        self.conv = nn.ModuleList([FeatureConversion(C * 2, inverse=bool(i % 2)) for i in range(num_dplayer)])
        self.embed = nn.Embedding(n_queries, cond_dim)
        self.film = nn.ModuleList()
        for i in range(num_dplayer):
            d = C * (2 if i % 2 else 1)
            f = nn.Linear(cond_dim, 2 * d)
            nn.init.zeros_(f.weight)                  # identity FiLM at init
            with torch.no_grad():
                f.bias.zero_()
                f.bias[:d] = 1.0
            self.film.append(f)

    def forward(self, mix, query):
        B, C, L0 = mix.shape
        dev = mix.device.type
        # pad so the number of STFT frames is even (needed by the rfft over time)
        padding = self.hop_length - L0 % self.hop_length
        if (L0 + padding) // self.hop_length % 2 == 0:
            padding += self.hop_length
        with torch.autocast(device_type=dev, enabled=False):
            x = F.pad(mix.float(), (0, padding))
            L = x.shape[-1]
            X = torch.stft(x.reshape(-1, L), window=self.window, return_complex=True, **self.stft_config)
            X = torch.view_as_real(X)                                     # (B*C, Fr, T, 2)
            X = X.permute(0, 3, 1, 2).reshape(B, C * 2, X.shape[1], X.shape[2])
            _, _, Fr, T = X.shape
            mean = X.mean(dim=(1, 2, 3), keepdim=True)
            std = X.std(dim=(1, 2, 3), keepdim=True)
            x = (X - mean) / (1e-5 + std)

        skips, lens, olens = deque(), deque(), deque()
        for sd in self.encoder:
            x, skip, l, ol = sd(x)
            skips.append(skip)
            lens.append(l)
            olens.append(ol)

        cond = self.embed(query)
        for dp, conv, film in zip(self.dp, self.conv, self.film):
            x = dp(x)
            g, b = film(cond).chunk(2, dim=-1)
            x = x * g[:, :, None, None] + b[:, :, None, None]
            x = conv(x)

        for fusion, su in self.decoder:
            x = fusion(x, skips.pop())
            x = su(x, lens.pop(), olens.pop())

        with torch.autocast(device_type=dev, enabled=False):
            x = x.float() * std + mean                                    # (B, 2C, Fr, T)
            x = x.reshape(B * C, 2, Fr, T).permute(0, 2, 3, 1)
            x = torch.view_as_complex(x.contiguous())
            y = torch.istft(x, window=self.window, length=L, **self.stft_config)
        return y.reshape(B, C, -1)[..., :L0]
