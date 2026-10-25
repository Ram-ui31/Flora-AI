"""Self-contained test: builds tiny fake MUSDB / MoisesDB / Slakh trees, then checks
the mixer's invariants. Needs no real data.

python tests/test_mixer.py
"""
import sys
import tempfile
from collections import Counter
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep import config  # noqa: E402
from sep.data import ConditionedMixDataset, MixerConfig, build_manifest  # noqa: E402

SR = config.SAMPLE_RATE
DUR = 8  # seconds per fake track


def tone(rng, mono=False, silent_first_half=False):
    t = np.arange(DUR * SR) / SR
    x = 0.2 * np.sin(2 * np.pi * rng.uniform(80, 2000) * t) + 0.02 * rng.standard_normal(t.size)
    if silent_first_half:
        x[: x.size // 2] = 0
    x = x.astype(np.float32)
    return x if mono else np.stack([x, x * 0.8], 1)


def make_fake(root: Path, rng):
    # MUSDB18-HQ
    for split, n in (("train", 6), ("test", 2)):
        for i in range(n):
            d = root / "musdb" / split / f"song{i}"
            d.mkdir(parents=True)
            for s in ("vocals", "drums", "bass", "other", "mixture"):
                sf.write(d / f"{s}.wav", tone(rng), SR)
    # MoisesDB (one track has two guitar files, one has an unknown folder)
    for i in range(8):
        d = root / "moises" / f"uuid{i}"
        for s in ("vocals", "drums", "guitar", "bass", "bowed_strings"):
            (d / s).mkdir(parents=True)
            sf.write(d / s / "a.wav", tone(rng, silent_first_half=(s == "guitar")), SR)
        if i == 0:
            sf.write(d / "guitar" / "b.wav", tone(rng), SR)
            (d / "weird_folder").mkdir()
            sf.write(d / "weird_folder" / "x.wav", tone(rng), SR)
    # Slakh (mono flac stems, includes a non-rendered stem)
    for split in ("train", "validation", "test"):
        for i in range(4):
            d = root / "slakh" / split / f"Track{i:05d}"
            (d / "stems").mkdir(parents=True)
            meta = {"stems": {
                "S00": {"inst_class": "Piano", "is_drum": False, "audio_rendered": True},
                "S01": {"inst_class": "Guitar", "is_drum": False, "audio_rendered": True},
                "S02": {"inst_class": "Drums", "is_drum": True, "audio_rendered": True},
                "S03": {"inst_class": "Bass", "is_drum": False, "audio_rendered": True},
                "S04": {"inst_class": "Strings", "is_drum": False, "audio_rendered": False},
            }}
            (d / "metadata.yaml").write_text(yaml.safe_dump(meta))
            for sid in ("S00", "S01", "S02", "S03"):
                sf.write(d / "stems" / f"{sid}.flac", tone(rng, mono=True), SR)


def main():
    rng = np.random.default_rng(0)
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        make_fake(root, rng)
        m = build_manifest(root / "musdb", root / "moises", root / "slakh")

        # --- manifest checks
        by_id = {t["id"]: t for t in m["tracks"]}
        mus = by_id["musdb/train/song0"]
        assert {s["label"] for s in mus["stems"]} == {"vocals", "drums", "other", "ambiguous"}
        assert all("S04" not in p for t in m["tracks"] if t["dataset"] == "slakh"
                   for s in t["stems"] for p in s["paths"]), "non-rendered stem indexed"
        g0 = [s for s in by_id["moisesdb/uuid0"]["stems"] if s["label"] == "guitar"][0]
        assert len(g0["paths"]) == 2, "multi-file stem not grouped"
        print(f"manifest ok: {len(m['tracks'])} tracks")

        # --- mixer checks
        cfg = MixerConfig(split="train", epoch_size=400, deterministic=True,
                          return_debug=True, segment_seconds=2.0)
        ds = ConditionedMixDataset(m, cfg)
        stats = Counter()
        for i in range(len(ds)):
            it = ds[i]
            mix, tgt, q = it["mix"], it["target"], config.QUERY_CLASSES[int(it["query"])]
            sums = it["sums"]
            assert mix.shape == tgt.shape == (2, int(2.0 * SR))
            assert torch.isfinite(mix).all() and mix.abs().max() <= cfg.peak_limit + 1e-5
            # mix is exactly the sum of what went in
            assert torch.allclose(mix, sum(sums.values()), atol=1e-5)
            if it["present"]:
                assert torch.allclose(tgt, sums[q], atol=1e-6)
            else:
                assert tgt.abs().max() == 0
            # a mix with an ambiguous stem must never be queried for what it may hide
            if sums["ambiguous"].abs().max() > 0:
                assert q in ("vocals", "drums"), f"ambiguous mix queried for {q}"
            stats[(q, bool(it["present"]))] += 1
            stats[it["mode"]] += 1
            stats[it["source"]] += 1

        # deterministic mode reproduces items
        assert torch.equal(ds[7]["mix"], ds[7]["mix"])
        absent = sum(v for k, v in stats.items() if isinstance(k, tuple) and not k[1])
        print("mixer ok:", dict(stats))
        print(f"absent fraction: {absent / len(ds):.2f} (p_absent={cfg.p_absent})")

        # DataLoader with workers works
        dl = torch.utils.data.DataLoader(
            ConditionedMixDataset(m, MixerConfig(epoch_size=16, segment_seconds=1.0)),
            batch_size=4, num_workers=2)
        b = next(iter(dl))
        assert b["mix"].shape == (4, 2, SR)
        print("dataloader ok:", tuple(b["mix"].shape))
    print("ALL TESTS PASSED")


if __name__ == "__main__":
    main()
