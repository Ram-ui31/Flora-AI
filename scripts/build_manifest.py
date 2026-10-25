"""Index the datasets once and write manifest.json.

python scripts/build_manifest.py \
    --musdb    /data/musdb18hq \
    --moisesdb /data/moisesdb/moisesdb_v0.1 \
    --slakh    /data/slakh2100_flac_redux \
    --out      manifest.json
Any dataset can be omitted.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep.data.sources import build_manifest, save_manifest, summarize  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--musdb")
ap.add_argument("--moisesdb")
ap.add_argument("--slakh")
ap.add_argument("--out", default="manifest.json")
a = ap.parse_args()

m = build_manifest(a.musdb, a.moisesdb, a.slakh)
save_manifest(m, a.out)
print(summarize(m))
print(f"\nwrote {a.out} ({len(m['tracks'])} tracks)")
