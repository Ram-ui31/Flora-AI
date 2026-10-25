"""On-the-fly conditioned mixing dataset.

Each item is (mix, query, target):
    mix    : (2, T) float32  - what the model hears
    query  : int             - index into config.QUERY_CLASSES
    target : (2, T) float32  - the stem of that class in the mix (zeros if absent)

Two ways to build a mix
-----------------------
coherent   : all stems of one real song at one time offset (musically realistic)
incoherent : stems drawn from *different* songs per label (unlimited variety)

Rules that matter
-----------------
* Absent queries: with prob p_absent the queried class is NOT in the mix and the
  target is silence. This teaches the model not to invent a guitar when asked.
* Ambiguous stems (MUSDB "other") may hide guitar/piano. A mix containing one is
  never used for a query it may hide, and is never used for an absent query of it.
* "present" is decided from the energy of the target *segment*, not from whether
  the song has that stem at all (stems are often silent for long stretches).
* Gains are applied per stem BEFORE summing, and any peak limiting scales mix and
  target together, so mix == target + rest always holds.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset

from .. import config


@dataclass
class MixerConfig:
    split: str = "train"
    segment_seconds: float = 6.0
    # dataset sampling weights (renormalised over datasets that can serve a request)
    dataset_weights: dict = field(default_factory=lambda: {
        "moisesdb": 0.4, "slakh": 0.35, "musdb": 0.25})
    # query class weights (default uniform -> class balanced)
    query_weights: dict | None = None
    p_coherent: float = 0.5
    p_absent: float = 0.15
    # incoherent mode: prob. that each NON-query label is included
    p_include: dict = field(default_factory=lambda: {
        "vocals": 0.6, "drums": 0.8, "guitar": 0.5, "piano": 0.5, "other": 0.9})
    max_other_stems: int = 2
    # hard negatives (run4): for piano/guitar queries, with this probability build an incoherent
    # mix that always contains the confusable partner (guitar<->piano) AND one sustained "other"
    # part (strings / pads / organ / synth / winds, config.SUSTAINED_OTHER).
    p_hard: float = 0.0
    # share of those hard examples taken from ONE song at an offset where guitar AND piano both
    # play (same key, tempo and chords - the real difficulty). The rest stay cross-song mixes.
    # Cross-song mixes are easier than real ones (Jeon et al. 2024, "Why does music source
    # separation benefit from cacophony?"), so on their own they under-train this case.
    p_hard_coh: float = 0.0
    hard_dataset_weights: dict | None = None   # dataset weights for same-song hard examples
    both_active_db: float = -35.0              # both guitar and piano must exceed this RMS
    hard_offset_retries: int = 12
    gain_db: tuple = (-6.0, 3.0)          # per-stem random gain
    p_channel_swap: float = 0.5           # per-stem L/R swap
    silence_db: float = -45.0             # target RMS below this = absent
    offset_retries: int = 6               # tries to land on an active target segment
    peak_limit: float = 0.95
    epoch_size: int = 20000
    deterministic: bool = False           # True for validation: item i is always the same
    seed: int = 0
    return_debug: bool = False            # also return per-label sums (tests / inspection)


def _rms_db(x: np.ndarray) -> float:
    return 10 * np.log10(np.mean(x ** 2) + 1e-12)


def _read(paths, start, frames) -> np.ndarray:
    """Sum of files, (2, frames), zero-padded on short reads, mono -> stereo."""
    out = np.zeros((config.CHANNELS, frames), dtype=np.float32)
    for p in paths:
        a, _ = sf.read(p, start=start, frames=frames, dtype="float32", always_2d=True)
        a = a.T
        if a.shape[0] == 1:
            a = np.repeat(a, 2, axis=0)
        a = a[:2]
        out[:, : a.shape[1]] += a
    return out


class ConditionedMixDataset(Dataset):
    def __init__(self, manifest: dict, cfg: MixerConfig | None = None):
        self.cfg = cfg or MixerConfig()
        c = self.cfg
        self.seg = int(round(c.segment_seconds * config.SAMPLE_RATE))
        self.tracks = [t for t in manifest["tracks"]
                       if t["split"] == c.split and c.dataset_weights.get(t["dataset"], 0) > 0]
        if not self.tracks:
            raise ValueError(f"no tracks for split={c.split!r}")
        self._ncalls = 0

        # clean stems per label per dataset: label -> ds -> [(ti, si)]
        self.by_label = {l: {} for l in config.USER_CLASSES}
        # coherent candidates: q -> ds -> [ti] ; separately for present / absent queries
        self.coh_present = {q: {} for q in config.QUERY_CLASSES}
        self.coh_absent = {q: {} for q in config.QUERY_CLASSES}
        self.sustained = {}          # ds -> [(ti, si)] of sustained "other" stems (hard negatives)
        self.coh_both = {}           # ds -> [ti] of songs with clean guitar AND piano stems

        for ti, t in enumerate(self.tracks):
            ds = t["dataset"]
            labels = set()
            blocked = set()
            for si, s in enumerate(t["stems"]):
                if s["label"] == "ambiguous":
                    blocked.update(s["may_contain"])
                else:
                    labels.add(s["label"])
                    self.by_label[s["label"]].setdefault(ds, []).append((ti, si))
                    if s["label"] == "other" and s.get("sub") in config.SUSTAINED_OTHER:
                        self.sustained.setdefault(ds, []).append((ti, si))
            for q in config.QUERY_CLASSES:
                if q in blocked:
                    continue
                (self.coh_present if q in labels else self.coh_absent)[q].setdefault(ds, []).append(ti)
            if {"guitar", "piano"} <= labels and not ({"guitar", "piano"} & blocked):
                self.coh_both.setdefault(ds, []).append(ti)

        qw = c.query_weights or {q: 1.0 for q in config.QUERY_CLASSES}
        self.qw = np.array([qw.get(q, 0.0) for q in config.QUERY_CLASSES], dtype=float)
        self.qw /= self.qw.sum()

    def __len__(self):
        return self.cfg.epoch_size

    # ------------------------------------------------------------ sampling
    def _rng(self, idx):
        if self.cfg.deterministic:
            return np.random.default_rng([self.cfg.seed, idx])
        self._ncalls += 1
        return np.random.default_rng([torch.initial_seed() % (2 ** 63), idx, self._ncalls])

    def _pick_ds(self, rng, pools: dict, weights: dict | None = None):
        weights = weights or self.cfg.dataset_weights
        names = [d for d, v in pools.items() if v and weights.get(d, 0) > 0]
        if not names:
            return None
        w = np.array([weights[d] for d in names], dtype=float)
        return names[rng.choice(len(names), p=w / w.sum())]

    def _offset(self, rng, frames):
        return int(rng.integers(0, max(1, frames - self.seg + 1)))

    def _coherent(self, rng, q, want_absent):
        pools = (self.coh_absent if want_absent else self.coh_present)[q]
        ds = self._pick_ds(rng, pools)
        if ds is None:
            return None
        t = self.tracks[pools[ds][rng.integers(len(pools[ds]))]]
        tgt = [s for s in t["stems"] if s["label"] == q]
        for _ in range(self.cfg.offset_retries):
            off = self._offset(rng, t["frames"])
            if want_absent:
                break
            if _rms_db(sum(_read(s["paths"], off, self.seg) for s in tgt)) > self.cfg.silence_db:
                break
        stems = [(s["label"], _read(s["paths"], off, self.seg)) for s in t["stems"]]
        return stems, ds

    def _coherent_both(self, rng, q, want_absent):
        """One song, one offset where guitar and piano are both audible. For an absent query
        the queried instrument's stems are dropped, so the mix still holds its partner (asked
        for guitar, only the song's piano is there -> the answer is silence)."""
        ds = self._pick_ds(rng, self.coh_both, self.cfg.hard_dataset_weights)
        if ds is None:
            return None
        t = self.tracks[self.coh_both[ds][rng.integers(len(self.coh_both[ds]))]]
        g = [s for s in t["stems"] if s["label"] == "guitar"]
        p = [s for s in t["stems"] if s["label"] == "piano"]
        for _ in range(self.cfg.hard_offset_retries):
            off = self._offset(rng, t["frames"])
            if all(_rms_db(sum(_read(s["paths"], off, self.seg) for s in grp)) > self.cfg.both_active_db
                   for grp in (g, p)):
                break
        stems = [(s["label"], _read(s["paths"], off, self.seg)) for s in t["stems"]
                 if not (want_absent and s["label"] == q)]
        return stems, ds

    def _draw_stem(self, rng, label, want_active, pools=None):
        pools = self.by_label[label] if pools is None else pools
        ds = self._pick_ds(rng, pools)
        if ds is None:
            return None
        pool = pools[ds]
        ti, si = pool[rng.integers(len(pool))]
        t = self.tracks[ti]
        paths = t["stems"][si]["paths"]
        for _ in range(self.cfg.offset_retries if want_active else 1):
            a = _read(paths, self._offset(rng, t["frames"]), self.seg)
            if not want_active or _rms_db(a) > self.cfg.silence_db:
                break
        return a

    def _incoherent(self, rng, q, want_absent, hard=False):
        stems = []
        partner = config.CONFUSABLE_PAIR.get(q) if hard else None
        for label in config.USER_CLASSES:
            if label == q:
                if want_absent:
                    continue
                n = 1
            elif label == partner:
                n = 1                                   # hard mix: partner always present
            else:
                if rng.random() >= self.cfg.p_include[label]:
                    continue
                n = int(rng.integers(1, self.cfg.max_other_stems + 1)) if label == "other" else 1
            for _ in range(n):
                a = self._draw_stem(rng, label, want_active=True)
                if a is not None:
                    stems.append((label, a))
        if hard and self.sustained:                     # plus one sustained "other" part
            a = self._draw_stem(rng, "other", want_active=True, pools=self.sustained)
            if a is not None:
                stems.append(("other", a))
        if not any(l != q for l, _ in stems):          # never return target alone / empty
            a = self._draw_stem(rng, "other", want_active=True)
            if a is not None:
                stems.append(("other", a))
        return stems, ("hard" if hard else "mixed")

    # ------------------------------------------------------------ item
    def __getitem__(self, idx):
        c = self.cfg
        rng = self._rng(idx)
        qi = int(rng.choice(len(config.QUERY_CLASSES), p=self.qw))
        q = config.QUERY_CLASSES[qi]
        want_absent = rng.random() < c.p_absent

        res = None
        if q in config.CONFUSABLE_PAIR and rng.random() < c.p_hard:
            if self.coh_both and rng.random() < c.p_hard_coh:
                mode = "hard_coh"
                res = self._coherent_both(rng, q, want_absent)
            if res is None:
                mode = "hard"
                res = self._incoherent(rng, q, want_absent, hard=True)
        else:
            mode = "coherent" if rng.random() < c.p_coherent else "incoherent"
            if mode == "coherent":
                res = self._coherent(rng, q, want_absent)
        if res is None:
            mode = "incoherent"
            res = self._incoherent(rng, q, want_absent)
        stems, source = res

        # per-stem augmentation, then group by label
        sums = {l: np.zeros((2, self.seg), np.float32) for l in config.STEM_LABELS}
        for label, a in stems:
            a = a * np.float32(10 ** (rng.uniform(*c.gain_db) / 20))
            if rng.random() < c.p_channel_swap:
                a = a[::-1]
            sums[label] += a

        mix = sum(sums.values())
        target = sums[q].copy()
        peak = float(np.abs(mix).max())
        if peak > c.peak_limit:
            g = np.float32(c.peak_limit / peak)
            mix *= g
            target *= g
            for l in sums:
                sums[l] *= g
        present = _rms_db(target) > c.silence_db
        if not present:
            target[:] = 0.0      # near-silent bleed is not a target; mix keeps it

        item = {
            "mix": torch.from_numpy(np.ascontiguousarray(mix)),
            "target": torch.from_numpy(np.ascontiguousarray(target)),
            "query": torch.tensor(qi),
            "present": torch.tensor(present),
            "mode": mode,
            "source": source,
        }
        if c.return_debug:
            item["sums"] = {l: torch.from_numpy(np.ascontiguousarray(v)) for l, v in sums.items()}
        return item
