"""Losses and metrics that handle both present and absent targets.

present target s:  thresholded negative SNR
    L = -10 log10( |s|^2 / (|s - s_hat|^2 + tau |s|^2) )      (caps at -30 dB for tau=1e-3)
absent target:     output energy relative to the mix
    L = 10 log10( |s_hat|^2 + tau |x|^2 ) - 10 log10( |x|^2 ) (floor at -30 dB)

SNR (not SI-SDR) is used on purpose: "other" = mix - sum(stems) needs correctly
SCALED stems, and SI-SDR ignores scale. Both terms are in dB so they share a range.
"""
import torch

EPS = 1e-8


def _energy(x):
    return x.pow(2).flatten(1).sum(-1)


def snr_loss(est, ref, tau=1e-3):
    num = _energy(ref)
    den = _energy(ref - est) + tau * num
    return -10 * torch.log10((num + EPS) / (den + EPS))


def absent_loss(est, mix, tau=1e-3):
    mix_e = _energy(mix)
    return 10 * torch.log10(_energy(est) + tau * mix_e + EPS) - 10 * torch.log10(mix_e + EPS)


def absent_leak(est, mix):
    """Fraction of the mix energy that leaks out on an absent query (linear, 0..~1).
    Linear on purpose: the dB version (absent_loss) has a gradient that grows without
    bound as the output gets quieter, which made run1 collapse to silence everywhere."""
    return _energy(est) / (_energy(mix) + EPS)


_WINDOWS = {}


def mrstft_loss(est, ref, resolutions=((512, 128), (1024, 256), (2048, 512)), eps=1e-4, under=0.0):
    """Multi-resolution STFT loss per item (B,): relative L1 on magnitudes + L1 on log
    magnitudes, averaged over resolutions. The log term weighs quiet passages as much as
    loud ones, which the waveform SNR does not (-> run2's level "pumping").
    under > 0: bins where the estimate is QUIETER than the reference cost (1 + under) times as
    much in the log term. Counters under-production (run3/run4: piano ~-9.5 dB, guitar ~-5.5 dB
    too quiet), which the SNR loss favours whenever the model is unsure."""
    B, C, T = est.shape
    e, r = est.reshape(B * C, T), ref.reshape(B * C, T)
    out = 0
    for n_fft, hop in resolutions:
        key = (n_fft, est.device)
        if key not in _WINDOWS:
            _WINDOWS[key] = torch.hann_window(n_fft, device=est.device)
        w = _WINDOWS[key]
        E = torch.stft(e, n_fft, hop, window=w, return_complex=True).abs().reshape(B, C, -1)
        R = torch.stft(r, n_fft, hop, window=w, return_complex=True).abs().reshape(B, C, -1)
        lin = (E - R).abs().flatten(1).sum(-1) / (R.flatten(1).sum(-1) + EPS)
        d = torch.log10(E + eps) - torch.log10(R + eps)
        log = d.abs().flatten(1).mean(-1)
        if under > 0:
            log = log + under * torch.relu(-d).flatten(1).mean(-1)
        out = out + lin + log
    return out / len(resolutions)


def conditioned_loss(est, target, mix, present, absent_weight=10.0, stft_weight=0.0, under=0.0):
    """Returns (per-item loss (B,), parts dict). Computed in fp32.
    present -> thresholded negative SNR (dB); absent -> absent_weight * leak fraction."""
    est, target, mix = est.float(), target.float(), mix.float()
    lp = snr_loss(est, target)
    ls = mrstft_loss(est, target, under=under) if stft_weight > 0 else torch.zeros_like(lp)
    la = absent_weight * absent_leak(est, mix)
    loss = torch.where(present, lp + stft_weight * ls, la)
    with torch.no_grad():
        parts = {
            "loss_present": lp[present].mean().item() if present.any() else float("nan"),
            "stft_present": ls[present].mean().item() if present.any() else float("nan"),
            "leak_absent": absent_leak(est, mix)[~present].mean().item() if (~present).any() else float("nan"),
        }
    return loss, parts


@torch.no_grad()
def sdr(est, ref):
    """Plain SDR in dB per item (no threshold), for reporting."""
    est, ref = est.float(), ref.float()
    return 10 * torch.log10((_energy(ref) + EPS) / (_energy(ref - est) + EPS))


@torch.no_grad()
def rel_energy_db(est, mix):
    """Output energy relative to the mix, dB. Lower is better on absent queries."""
    est, mix = est.float(), mix.float()
    return 10 * torch.log10((_energy(est) + EPS) / (_energy(mix) + EPS))
