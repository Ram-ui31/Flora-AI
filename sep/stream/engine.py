"""Streaming separation: per-listener buffering + overlap-add, and a shared batched model worker.

Timeline (per listener)
-----------------------
Audio arrives in arbitrary-sized pieces. Internally it lives on a "padded timeline" that starts
with P = W - H zeros (pre-roll), so the first window can run as soon as H real samples arrived.

    window k covers padded positions [k*H, k*H + W)        (W = window, H = hop, W = K*H)

Each window's estimate is multiplied by a taper and added to an accumulator; the taper sum is
accumulated too. When window k has been added, positions [k*H, k*H + H) will never be touched
again, so they are FINAL: they are divided by their taper sum and emitted. With the pre-roll
dropped, emitted output lines up sample-for-sample with the input.

Delay a listener hears for an input sample: between W - H and W seconds (it has to wait for the
window that completes it), plus compute time and network.

Instrument switching: on a change, the next window is run for both the old and the new
instrument and the emitted hop is cross-faded old -> new (one hop long), so there is no click.
"other" = mix minus the query stems in `other_from` (default: all four). A demo that hides guitar
uses other_from = (vocals, drums, piano), so guitar lands in "other" instead of disappearing.

The same math as sep.inference.separate (Hann taper clamped at 1e-3, normalised by the taper
sum); tests/test_stream.py checks identity reconstruction and equality with offline separation.
"""
from __future__ import annotations

import asyncio
import concurrent.futures
import time
from dataclasses import dataclass, field

import numpy as np
import torch

from .. import config


def taper(n: int) -> np.ndarray:
    return np.clip(np.hanning(n), 1e-3, None).astype(np.float32)


class _Acc:
    """Overlap-add accumulator for one query on one listener."""

    def __init__(self, W: int, C: int = 2):
        self.out = np.zeros((C, W), np.float32)
        self.norm = np.zeros(W, np.float32)

    def add_and_pop(self, est: np.ndarray, tap: np.ndarray, H: int) -> np.ndarray:
        self.out += est * tap
        self.norm += tap
        done = self.out[:, :H] / self.norm[:H]
        self.out = np.concatenate([self.out[:, H:], np.zeros_like(self.out[:, :H])], 1)
        self.norm = np.concatenate([self.norm[H:], np.zeros(H, np.float32)])
        return done


@dataclass
class Hop:
    """One emitted piece of output."""
    seq: int            # 0, 1, 2, ... (consecutive; a gap would mean lost audio)
    start: int          # index of its first sample in the listener's INPUT stream
    stem: np.ndarray    # (C, n) the chosen instrument
    rest: np.ndarray    # (C, n) input minus stem
    instrument: str


class StreamState:
    """Everything one listener needs. No I/O and no model inside: windows go out through
    `next_window()` and results come back through `push_result()`, so the same object is used
    by the server, the load test and the unit tests."""

    def __init__(self, window_s: float, overlap: float = 0.5, instrument: str = "vocals",
                 sr: int = config.SAMPLE_RATE, channels: int = 2, allowed=None):
        self.W = int(round(window_s * sr))
        self.H = int(round(self.W * (1 - overlap)))
        assert self.W % self.H == 0, "window must be a whole number of hops"
        self.K = self.W // self.H
        self.C = channels
        self.sr = sr
        self.allowed = list(allowed) if allowed else list(config.USER_CLASSES)
        if instrument not in self.allowed:
            raise ValueError(f"unknown instrument {instrument!r}")
        self.P = self.W - self.H                         # pre-roll
        self.tap = taper(self.W)
        self.hist = np.zeros((channels, self.P), np.float32)   # last W-H samples of the timeline
        self.pending = np.zeros((channels, 0), np.float32)     # received, not yet windowed
        self.instrument = instrument
        self._next_instr = instrument
        self.acc: dict[str, _Acc] = {instrument: _Acc(self.W, channels)}
        self.k = 0                                       # windows completed
        self.seq = 0
        self.received = 0                                # input samples received so far
        self._emit_pos = -self.P                         # input index of the next emitted sample

    # ------------------------------------------------------------- input side
    def feed(self, pcm: np.ndarray):
        """pcm: (C, n) float32."""
        self.pending = np.concatenate([self.pending, pcm.astype(np.float32, copy=False)], 1)
        self.received += pcm.shape[1]

    def select(self, instrument: str):
        if instrument not in self.allowed:
            raise ValueError(f"unknown instrument {instrument!r}")
        self._next_instr = instrument

    def backlog_s(self) -> float:
        """Received audio still waiting to be windowed (grows if inference falls behind)."""
        return self.pending.shape[1] / self.sr

    def next_window(self):
        """Returns (window (C, W), instruments to run) or None if not enough audio yet.
        `instruments` has two entries during an instrument switch (old, new)."""
        if self.pending.shape[1] < self.H:
            return None
        new = self.pending[:, :self.H]
        self.pending = self.pending[:, self.H:]
        win = np.concatenate([self.hist, new], 1)
        self.hist = win[:, self.H:]
        self._win = win
        if self._next_instr != self.instrument:
            self.acc.setdefault(self._next_instr, _Acc(self.W, self.C))
            return win, [self.instrument, self._next_instr]
        return win, [self.instrument]

    # ------------------------------------------------------------- output side
    def push_result(self, ests: dict[str, np.ndarray]) -> Hop | None:
        """ests: instrument -> (C, W) estimate for the window from next_window().
        Returns the emitted Hop (None while the pre-roll is being flushed)."""
        H = self.H
        outs = {name: self.acc[name].add_and_pop(ests[name], self.tap, H) for name in ests}
        mix_hop = self._win[:, :H]
        if len(outs) == 2:                                   # cross-fade old -> new over one hop
            old, new = self.instrument, self._next_instr
            r = np.linspace(0, 1, H, dtype=np.float32)
            stem = outs[old] * (1 - r) + outs[new] * r
            del self.acc[old]
            self.instrument = new
            self.acc[new] = self.acc[new]
        else:
            stem = outs[self.instrument]
        self.k += 1
        start = self._emit_pos
        self._emit_pos += H
        if start + H <= 0:                                   # entirely pre-roll: nothing to emit
            return None
        if start < 0:                                        # partly pre-roll (only if P % H != 0)
            stem, mix_hop, start = stem[:, -start:], mix_hop[:, -start:], 0
        hop = Hop(self.seq, start, stem, mix_hop - stem, self.instrument)
        self.seq += 1
        return hop


def model_queries(instruments, other_from=config.QUERY_CLASSES):
    """Model calls needed: 'other' needs every stem in other_from."""
    qs = []
    for name in instruments:
        for q in (other_from if name == "other" else [name]):
            if q not in qs:
                qs.append(q)
    return qs


def assemble(instruments, window, q_out, other_from=config.QUERY_CLASSES):
    """Turn per-query model outputs into per-instrument estimates."""
    ests = {}
    for name in instruments:
        ests[name] = window - sum(q_out[q] for q in other_from) if name == "other" else q_out[name]
    return ests


# ----------------------------------------------------------------- shared model worker
@dataclass
class _Req:
    window: np.ndarray
    query: int
    fut: asyncio.Future
    t_enq: float = field(default_factory=time.perf_counter)


class InferenceWorker:
    """One model, many listeners. Requests from all connections go into one queue; the worker
    takes up to `max_batch` at a time (any mix of instruments - the model is conditioned per item),
    runs them in a background thread so the event loop keeps receiving/sending audio, and resolves
    each request's future. Nothing is ever dropped: if the model is slower than real time, queues
    grow and listeners see their backlog / delay rise (reported in stats)."""

    def __init__(self, model, device, max_batch=16):
        self.model, self.device, self.max_batch = model, device, max_batch
        self.q: asyncio.Queue[_Req] = asyncio.Queue()
        self.pool = concurrent.futures.ThreadPoolExecutor(max_workers=1)
        self.stats = {"batches": 0, "items": 0, "busy_s": 0.0, "max_batch_seen": 0}
        self._task = None

    def start(self):
        self._task = asyncio.get_running_loop().create_task(self._run())

    async def infer(self, window: np.ndarray, query: str) -> np.ndarray:
        fut = asyncio.get_running_loop().create_future()
        await self.q.put(_Req(window, config.QUERY_INDEX[query], fut))
        return await fut

    def _forward(self, wins, qs):
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(wins)).to(self.device)
            q = torch.tensor(qs, dtype=torch.long, device=self.device)
            y = self.model(x, q)
            return y.float().cpu().numpy()

    async def _run(self):
        loop = asyncio.get_running_loop()
        while True:
            reqs = [await self.q.get()]
            while len(reqs) < self.max_batch and not self.q.empty():
                reqs.append(self.q.get_nowait())
            t0 = time.perf_counter()
            try:
                y = await loop.run_in_executor(self.pool, self._forward,
                                               [r.window for r in reqs], [r.query for r in reqs])
                for r, out in zip(reqs, y):
                    if not r.fut.done():
                        r.fut.set_result(out)
            except Exception as e:                          # never leave a listener hanging
                for r in reqs:
                    if not r.fut.done():
                        r.fut.set_exception(e)
            dt = time.perf_counter() - t0
            s = self.stats
            s["batches"] += 1
            s["items"] += len(reqs)
            s["busy_s"] += dt
            s["max_batch_seen"] = max(s["max_batch_seen"], len(reqs))


async def process_window(state: StreamState, worker: InferenceWorker,
                         other_from=config.QUERY_CLASSES) -> Hop | None:
    """Run the next ready window of one listener through the shared worker."""
    nxt = state.next_window()
    if nxt is None:
        return None
    win, instruments = nxt
    qs = model_queries(instruments, other_from)
    outs = await asyncio.gather(*(worker.infer(win, q) for q in qs))
    return state.push_result(assemble(instruments, win, dict(zip(qs, outs)), other_from))
