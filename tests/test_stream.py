"""Streaming engine tests:  python -m pytest -q tests/test_stream.py

1. identity model -> the streamed stem equals the input exactly (overlap-add is lossless,
   no clicks at window seams), for every window/overlap and any input piece size;
2. real (small, random) SCNet -> streaming == offline separate() on the same windows;
3. instrument switch -> output continuous, ends on the new instrument; "other" = mix - 4 stems;
4. the async server path (InferenceWorker, several listeners batched together) gives the
   same output as the synchronous path.
"""
import asyncio
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep import config  # noqa: E402
from sep.inference import separate  # noqa: E402
from sep.models import build_model  # noqa: E402
from sep.stream.engine import (InferenceWorker, StreamState, assemble, model_queries,  # noqa: E402
                               process_window)

SR = config.SAMPLE_RATE


def run_sync(state, x, fn, piece):
    """Feed x (C, T) in pieces of `piece` samples; fn(window, query) -> estimate."""
    hops = []
    for i in range(0, x.shape[1], piece):
        state.feed(x[:, i:i + piece])
        while (nxt := state.next_window()) is not None:
            win, instruments = nxt
            qs = model_queries(instruments)
            outs = {q: fn(win, q) for q in qs}
            h = state.push_result(assemble(instruments, win, outs))
            if h is not None:
                hops.append(h)
    return hops


def joined(hops, key="stem"):
    for a, b in zip(hops, hops[1:]):
        assert b.seq == a.seq + 1 and b.start == a.start + a.stem.shape[1]   # no gaps
    return np.concatenate([getattr(h, key) for h in hops], 1)


@pytest.mark.parametrize("window_s,overlap,piece", [(0.5, 0.5, 777), (1.0, 0.5, 4410), (0.4, 0.75, 1000)])
def test_identity_reconstruction(window_s, overlap, piece):
    rng = np.random.default_rng(0)
    x = rng.standard_normal((2, int(3.3 * SR))).astype(np.float32)
    st = StreamState(window_s, overlap, "vocals")
    hops = run_sync(st, x, lambda w, q: w.copy(), piece)
    y = joined(hops)
    assert hops[0].start == 0
    n = y.shape[1]
    np.testing.assert_allclose(y, x[:, :n], atol=1e-5)
    np.testing.assert_allclose(joined(hops, "rest"), 0, atol=1e-5)
    # output trails the input by the pre-roll (W - H, the streaming delay) plus < one hop
    assert x.shape[1] - st.P - st.H < n <= x.shape[1] - st.P


def small_model():
    torch.manual_seed(0)
    return build_model({"arch": "scnet", "scnet_width": 32, "layers": 2}).eval()


def offline_equivalent(model, x, q, st):
    """Offline separate() on the same padded timeline (pre-roll in front)."""
    xp = torch.from_numpy(np.pad(x, ((0, 0), (st.P, 0))))
    y = separate(model, xp, q, window_s=st.W / SR, overlap=1 - st.H / st.W)
    return y.numpy()[:, st.P:]


def test_matches_offline():
    model = small_model()
    rng = np.random.default_rng(1)
    x = (0.1 * rng.standard_normal((2, int(2.6 * SR)))).astype(np.float32)
    st = StreamState(0.5, 0.5, "guitar")

    def fn(w, q):
        with torch.no_grad():
            return model(torch.from_numpy(w)[None], torch.tensor([config.QUERY_INDEX[q]]))[0].numpy()

    y = joined(run_sync(st, x, fn, 3000))
    ref = offline_equivalent(model, x, "guitar", st)
    n = y.shape[1] - st.H              # offline pads the tail differently; compare the settled part
    np.testing.assert_allclose(y[:, :n], ref[:, :n], atol=2e-4)


def test_switch_and_other():
    rng = np.random.default_rng(2)
    x = rng.standard_normal((2, int(3 * SR))).astype(np.float32)
    # fake model: each query returns a fixed fraction of the window -> stems are scaled mixes
    frac = {"vocals": 0.1, "drums": 0.2, "guitar": 0.3, "piano": 0.15}
    st = StreamState(0.5, 0.5, "vocals")
    hops = []
    for i in range(0, x.shape[1], 2205):
        if i == 2205 * 20:
            st.select("guitar")
        if i == 2205 * 40:
            st.select("other")
        hops += run_sync_step(st, x[:, i:i + 2205], frac)
    y = joined(hops)
    n = y.shape[1]
    t_g = next(h for h in hops if h.instrument == "guitar").start
    t_o = next(h for h in hops if h.instrument == "other").start
    H = st.H
    # before the first switch: vocals; well after: guitar; after the second: other = 1 - sum(frac)
    np.testing.assert_allclose(y[:, :t_g], 0.1 * x[:, :t_g], atol=1e-5)
    np.testing.assert_allclose(y[:, t_g + H:t_o], 0.3 * x[:, t_g + H:t_o], atol=1e-5)
    np.testing.assert_allclose(y[:, t_o + H:n], (1 - sum(frac.values())) * x[:, t_o + H:n], atol=1e-5)
    # cross-fade hop lies between the two levels and is continuous at both ends (no click)
    seg = y[:, t_g:t_g + H] / np.where(np.abs(x[:, t_g:t_g + H]) > 1e-3, x[:, t_g:t_g + H], np.nan)
    g = np.nanmedian(seg[:, :50]), np.nanmedian(seg[:, -50:])
    assert abs(g[0] - 0.1) < 0.02 and abs(g[1] - 0.3) < 0.02


def run_sync_step(st, piece, frac):
    st.feed(piece)
    out = []
    while (nxt := st.next_window()) is not None:
        win, instruments = nxt
        qs = model_queries(instruments)
        h = st.push_result(assemble(instruments, win, {q: frac[q] * win for q in qs}))
        if h is not None:
            out.append(h)
    return out


def test_async_worker_matches_sync():
    model = small_model()
    rng = np.random.default_rng(3)
    xs = [(0.1 * rng.standard_normal((2, int(1.5 * SR)))).astype(np.float32) for _ in range(3)]
    instr = ["vocals", "piano", "other"]

    async def main():
        worker = InferenceWorker(model, "cpu", max_batch=8)
        worker.start()

        async def listener(x, name):
            st = StreamState(0.5, 0.5, name)
            hops = []
            for i in range(0, x.shape[1], 4000):
                st.feed(x[:, i:i + 4000])
                while (h := await process_window(st, worker)) is not None or st.pending.shape[1] >= st.H:
                    if h is not None:
                        hops.append(h)
            return joined(hops)

        res = await asyncio.gather(*(listener(x, n) for x, n in zip(xs, instr)))
        return res, worker.stats

    res, stats = asyncio.run(main())
    assert stats["max_batch_seen"] > 1                      # listeners really were batched together

    def fn(w, q):
        with torch.no_grad():
            return model(torch.from_numpy(w)[None], torch.tensor([config.QUERY_INDEX[q]]))[0].numpy()

    for x, name, y in zip(xs, instr, res):
        ref = joined(run_sync(StreamState(0.5, 0.5, name), x, fn, 4000))
        np.testing.assert_allclose(y, ref, atol=1e-4)


def test_other_without_guitar():
    """Demo set (no guitar): other = mix - vocals - drums - piano, so guitar ends up in 'other'."""
    rng = np.random.default_rng(4)
    x = rng.standard_normal((2, int(2 * SR))).astype(np.float32)
    frac = {"vocals": 0.1, "drums": 0.2, "guitar": 0.3, "piano": 0.15}
    of = ("vocals", "drums", "piano")
    st = StreamState(0.5, 0.5, "other", allowed=["vocals", "drums", "piano", "other"])
    hops = []
    for i in range(0, x.shape[1], 3000):
        st.feed(x[:, i:i + 3000])
        while (nxt := st.next_window()) is not None:
            win, instruments = nxt
            qs = model_queries(instruments, of)
            assert "guitar" not in qs
            h = st.push_result(assemble(instruments, win, {q: frac[q] * win for q in qs}, of))
            if h is not None:
                hops.append(h)
    y = joined(hops)
    np.testing.assert_allclose(y, (1 - 0.45) * x[:, :y.shape[1]], atol=1e-5)
    with pytest.raises(ValueError):
        st.select("guitar")
