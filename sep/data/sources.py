"""Index MUSDB18-HQ, MoisesDB and Slakh2100 into one manifest.

Every stem is mapped to one label from config.STEM_LABELS. The manifest is a
plain JSON file so indexing (which reads every audio header) runs only once.

Manifest format
---------------
{
  "sample_rate": 44100,
  "tracks": [
    {"id": "musdb/train/Some Song", "dataset": "musdb", "split": "train",
     "frames": 9876543,
     "stems": [{"label": "vocals", "paths": ["/abs/vocals.wav"], "may_contain": []},
               ...]}
  ]
}
"""
from __future__ import annotations

import hashlib
import json
import os
import warnings
from pathlib import Path

import soundfile as sf

from .. import config

AUDIO_EXT = (".wav", ".flac")


# ----------------------------------------------------------------- helpers
def _frames(path: Path) -> int:
    info = sf.info(str(path))
    if info.samplerate != config.SAMPLE_RATE:
        raise ValueError(f"{path}: sample rate {info.samplerate} != {config.SAMPLE_RATE}")
    return info.frames


def _hash_split(key: str, val_fraction: float) -> str:
    """Deterministic train/valid split for datasets without an official one."""
    h = int(hashlib.md5(key.encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
    return "valid" if h < val_fraction else "train"


def _track(id_, dataset, split, stems):
    """stems: list of (label, [paths], may_contain, sub). `sub` = the dataset's own instrument
    name (MoisesDB folder, MUSDB stem, Slakh inst_class) - used for hard negatives / diagnostics."""
    stems = [s for s in stems if s[1]]
    if not stems:
        return None
    try:
        # aligned multitracks: usable length = shortest file
        frames = min(_frames(p) for st in stems for p in st[1])
    except (RuntimeError, ValueError) as e:
        warnings.warn(f"skipping {id_}: {e}")
        return None
    return {
        "id": id_,
        "dataset": dataset,
        "split": split,
        "frames": int(frames),
        "stems": [
            {"label": lab, "paths": [str(p) for p in paths], "may_contain": list(mc), "sub": sub}
            for lab, paths, mc, sub in stems
        ],
    }


# ------------------------------------------------------------------ MUSDB
def index_musdb(root: str | os.PathLike, val_fraction: float = 0.1) -> list[dict]:
    """root = folder containing train/ and test/ (MUSDB18-HQ, wav)."""
    root = Path(root).expanduser().resolve()
    tracks = []
    for official in ("train", "test"):
        d = root / official
        if not d.is_dir():
            continue
        for song in sorted(p for p in d.iterdir() if p.is_dir()):
            split = "test" if official == "test" else _hash_split(song.name, val_fraction)
            stems = []
            for name, (label, may_contain) in config.MUSDB_MAP.items():
                f = song / f"{name}.wav"
                stems.append((label, [f] if f.exists() else [], may_contain, name))
            t = _track(f"musdb/{official}/{song.name}", "musdb", split, stems)
            if t:
                tracks.append(t)
    return tracks


# --------------------------------------------------------------- MoisesDB
def index_moisesdb(root: str | os.PathLike, val_fraction: float = 0.1,
                   test_fraction: float = 0.1) -> list[dict]:
    """root = folder whose sub-folders are track uuids (e.g. moisesdb_v0.1/)."""
    root = Path(root).expanduser().resolve()
    tracks = []
    unknown = set()
    for tdir in sorted(p for p in root.iterdir() if p.is_dir()):
        stems = []
        for sdir in sorted(p for p in tdir.iterdir() if p.is_dir()):
            label = config.MOISESDB_MAP.get(sdir.name)
            if label is None:
                unknown.add(sdir.name)
                label = "other"
            files = sorted(f for f in sdir.iterdir() if f.suffix.lower() in AUDIO_EXT)
            stems.append((label, files, [], sdir.name))
        # split: test first, then valid, from independent hashes
        if _hash_split("test:" + tdir.name, test_fraction) == "valid":
            split = "test"
        else:
            split = _hash_split(tdir.name, val_fraction)
        t = _track(f"moisesdb/{tdir.name}", "moisesdb", split, stems)
        if t:
            tracks.append(t)
    if unknown:
        warnings.warn(f"MoisesDB: unknown stem folders mapped to 'other': {sorted(unknown)}")
    return tracks


# ----------------------------------------------------------------- Slakh
def index_slakh(root: str | os.PathLike) -> list[dict]:
    """root = slakh2100_flac_redux/ containing train/ validation/ test/."""
    import yaml

    root = Path(root).expanduser().resolve()
    split_names = {"train": "train", "validation": "valid", "test": "test"}
    tracks = []
    for folder, split in split_names.items():
        d = root / folder
        if not d.is_dir():
            continue
        for tdir in sorted(p for p in d.iterdir() if p.is_dir()):
            meta_f = tdir / "metadata.yaml"
            if not meta_f.exists():
                continue
            meta = yaml.safe_load(meta_f.read_text())
            stems = []
            for sid, s in (meta.get("stems") or {}).items():
                if not s.get("audio_rendered", True):
                    continue
                f = next((tdir / "stems" / f"{sid}{ext}" for ext in AUDIO_EXT
                          if (tdir / "stems" / f"{sid}{ext}").exists()), None)
                if f is None:
                    continue
                if s.get("is_drum"):
                    label = "drums"
                else:
                    label = config.SLAKH_MAP.get(s.get("inst_class"), config.SLAKH_DEFAULT)
                stems.append((label, [f], [], "Drums" if s.get("is_drum") else s.get("inst_class", "?")))
            t = _track(f"slakh/{folder}/{tdir.name}", "slakh", split, stems)
            if t:
                tracks.append(t)
    return tracks


# --------------------------------------------------------------- manifest
def build_manifest(musdb: str | None = None, moisesdb: str | None = None,
                   slakh: str | None = None) -> dict:
    tracks = []
    if musdb:
        tracks += index_musdb(musdb)
    if moisesdb:
        tracks += index_moisesdb(moisesdb)
    if slakh:
        tracks += index_slakh(slakh)
    return {"sample_rate": config.SAMPLE_RATE, "tracks": tracks}


def save_manifest(manifest: dict, path: str | os.PathLike) -> None:
    Path(path).write_text(json.dumps(manifest, indent=1))


def load_manifest(path: str | os.PathLike) -> dict:
    m = json.loads(Path(path).read_text())
    assert m["sample_rate"] == config.SAMPLE_RATE, "manifest sample rate mismatch"
    return m


def summarize(manifest: dict) -> str:
    """Hours of clean audio per (dataset, split, label) — useful sanity check."""
    from collections import defaultdict
    sr = manifest["sample_rate"]
    acc = defaultdict(float)
    ntracks = defaultdict(int)
    for t in manifest["tracks"]:
        ntracks[(t["dataset"], t["split"])] += 1
        for s in t["stems"]:
            acc[(t["dataset"], t["split"], s["label"])] += t["frames"] / sr / 3600
    lines = ["dataset   split  tracks  " + "  ".join(f"{l:>9}" for l in config.STEM_LABELS)]
    for (ds, sp), n in sorted(ntracks.items()):
        hrs = "  ".join(f"{acc[(ds, sp, l)]:9.2f}" for l in config.STEM_LABELS)
        lines.append(f"{ds:9} {sp:6} {n:6}  {hrs}")
    lines.append("(hours of stem audio per label; a track with 2 guitar stems counts twice)")
    return "\n".join(lines)
