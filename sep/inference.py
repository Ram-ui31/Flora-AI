"""Offline inference with windowed overlap-add (the same mechanics the streaming
server uses, just over a whole file at once).

    model  = load_model("checkpoints/scnet2/best.pt")
    stems  = separate_all(model, mix)       # dict: vocals, drums, guitar, piano, other
    stem   = separate(model, mix, "guitar")

Overlap-add: the mix is cut into windows of `window_s` seconds advancing by
`hop = window * (1 - overlap)`. Each window's estimate is multiplied by a Hann
taper and summed; the result is divided by the summed tapers, so any overlap ratio
reconstructs at unity gain with smooth (click-free) seams.
"""
from __future__ import annotations

import torch

from . import config
from .models import build_model


def load_model(ckpt_path, device="cpu"):
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    a = dict(ck["args"], grad_checkpoint=False)
    model = build_model(a, n_queries=len(ck.get("query_classes", config.QUERY_CLASSES)))
    model.load_state_dict(ck["model"])
    return model.to(device).eval()


def _taper(n, device):
    w = torch.hann_window(n, periodic=False, device=device)
    return w.clamp_min(1e-3)          # never exactly 0, so edges still normalise


@torch.no_grad()
def separate(model, mix, target, window_s=4.0, overlap=0.5, batch=8):
    """mix: (2, T) float tensor at config.SAMPLE_RATE. target: class name or index."""
    q = config.QUERY_INDEX[target] if isinstance(target, str) else int(target)
    dev = next(model.parameters()).device
    mix = mix.to(dev)
    C, T = mix.shape
    W = int(window_s * config.SAMPLE_RATE)
    H = max(1, int(W * (1 - overlap)))
    n = max(1, -(-(T - W) // H) + 1) if T > W else 1
    total = (n - 1) * H + W
    x = torch.nn.functional.pad(mix, (0, total - T))
    starts = [i * H for i in range(n)]
    taper = _taper(W, dev)
    out = torch.zeros(C, total, device=dev)
    norm = torch.zeros(total, device=dev)
    for i in range(0, n, batch):
        s = starts[i:i + batch]
        chunk = torch.stack([x[:, a:a + W] for a in s])
        qs = torch.full((len(s),), q, dtype=torch.long, device=dev)
        est = model(chunk, qs)
        for j, a in enumerate(s):
            out[:, a:a + W] += est[j] * taper
            norm[a:a + W] += taper
    return (out / norm)[:, :T]


@torch.no_grad()
def separate_all(model, mix, **kw):
    """All 4 query stems + other = mix - sum(stems). The 5 outputs sum to the mix."""
    stems = {c: separate(model, mix, c, **kw) for c in config.QUERY_CLASSES}
    stems["other"] = mix.to(stems["vocals"].device) - sum(stems.values())
    return stems
