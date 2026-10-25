"""Full-track evaluation on held-out songs, using the same overlap-add inference
as deployment.

python scripts/evaluate.py --ckpt checkpoints/run2/best.pt --manifest manifest_eval.json \
       --split test --out eval/run2_test.json

Per (track, query):
  present target -> SDR of the estimate, plus SDRi = SDR(est) - SDR(mix as estimate)
  absent target  -> leak: output energy relative to the mix, dB (lower is better)
Ambiguous stems (MUSDB "other") block guitar/piano queries on that track, exactly as in training.
Also reports the "other" residual (mix - sum of 4 stems) where the dataset labels it cleanly.
"""
import argparse
import json
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep import config  # noqa: E402
from sep.data import load_manifest  # noqa: E402
from sep.inference import load_model, separate  # noqa: E402

SILENT_DB = -45.0


def db(x):
    return 10 * np.log10(x + 1e-12)


def sdr(est, ref):
    return float(db(np.sum(ref ** 2)) - db(np.sum((ref - est) ** 2)))


def diagnostics(est, sums, q, sr=44100, other_split=None):
    """Listening-test proxies.
    gain_*: per 100 ms, how much of the TRUE stem comes through (projection gain, dB);
            median ~0 dB is right; a wide p10..p90 spread / many dips = audible "pumping".
    drums_leak_db: energy of the true drums inside this estimate, relative to the wanted
            component (least squares over all true stems, 1 s frames). Lower = less drums."""
    ref = sums[q]
    H = sr // 10
    g = []
    for s0 in range(0, ref.shape[1] - H, H):
        r, e = ref[:, s0:s0 + H].reshape(-1), est[:, s0:s0 + H].reshape(-1)
        rr = r.dot(r)
        if db(rr / r.size) < SILENT_DB:
            continue
        a = e.dot(r) / rr
        g.append(db(a * a) if a > 0 else -60.0)
    out = {}
    if g:
        g = np.clip(np.array(g), -60, 20)
        med = float(np.median(g))
        out.update(gain_median_db=med, gain_spread_db=float(np.percentile(g, 90) - np.percentile(g, 10)),
                   dip_frac=float(np.mean(g < med - 6)))
    # Projection of the estimate onto the true sources, 1 s frames. "other" is split into its
    # sustained parts (strings/pads/organ/synth/winds; config.SUSTAINED_OTHER) and the rest.
    groups = {l: sums[l] for l in ("vocals", "drums", "guitar", "piano")}
    if other_split is not None:
        groups["other_sustained"], groups["other_rest"] = other_split
    else:
        groups["other"] = sums["other"]
    names = [k for k, v in groups.items() if db(np.mean(v ** 2)) > SILENT_DB]
    if q in names and len(names) > 1:
        F = sr
        parts = np.zeros(len(names))
        for s0 in range(0, ref.shape[1] - F + 1, F):
            Rm = np.stack([groups[k][:, s0:s0 + F].reshape(-1) for k in names], 1)
            coef, *_ = np.linalg.lstsq(Rm, est[:, s0:s0 + F].reshape(-1), rcond=None)
            parts += (coef ** 2) * (Rm ** 2).sum(0)
        rel = lambda k: float(db(parts[names.index(k)]) - db(parts[names.index(q)]))  # noqa: E731
        if q != "drums" and "drums" in names:
            out["drums_leak_db"] = rel("drums")
        partner = config.CONFUSABLE_PAIR.get(q)
        if partner in names:
            out["partner_leak_db"] = rel(partner)          # guitar inside piano / piano inside guitar
        if "other_sustained" in names:
            out["sustained_leak_db"] = rel("other_sustained")   # strings/pads/organ/synth inside stem
    return out


def load_stem(paths, frames):
    out = np.zeros((2, frames), np.float32)
    for p in paths:
        a, _ = sf.read(p, frames=frames, dtype="float32", always_2d=True)
        a = a.T
        if a.shape[0] == 1:
            a = np.repeat(a, 2, 0)
        out[:, : a.shape[1]] += a[:2]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--manifest", default="manifest_eval.json")
    ap.add_argument("--split", default="test")
    ap.add_argument("--datasets", default="moisesdb,musdb")
    ap.add_argument("--window", type=float, default=4.0)
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--out", default="eval/results.json")
    ap.add_argument("--limit", type=int, default=0, help="max tracks per dataset (0 = all)")
    a = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, dev)
    m = load_manifest(a.manifest)
    wanted = set(a.datasets.split(","))
    tracks = [t for t in m["tracks"] if t["split"] == a.split and t["dataset"] in wanted]
    if a.limit:
        per = {}
        tracks = [t for t in tracks if per.setdefault(t["dataset"], []).append(1) or len(per[t["dataset"]]) <= a.limit]
    rows = []
    t0 = time.time()
    audio_s = 0.0
    for ti, t in enumerate(tracks):
        frames = t["frames"]
        audio_s += frames / config.SAMPLE_RATE
        sums = {l: np.zeros((2, frames), np.float32) for l in config.STEM_LABELS}
        blocked = set()
        other_sus = np.zeros((2, frames), np.float32)
        other_rest = np.zeros((2, frames), np.float32)
        has_sub = all("sub" in s for s in t["stems"])
        for s in t["stems"]:
            stem_audio = load_stem(s["paths"], frames)
            sums[s["label"]] += stem_audio
            if s["label"] == "other":
                if s.get("sub") in config.SUSTAINED_OTHER:
                    other_sus += stem_audio
                else:
                    other_rest += stem_audio
            if s["label"] == "ambiguous":
                blocked.update(s["may_contain"])
        mix = sum(sums.values())
        mix_t = torch.from_numpy(mix)
        ests = {}
        for q in config.QUERY_CLASSES:
            if q in blocked:
                continue
            est = separate(model, mix_t, q, window_s=a.window, overlap=a.overlap).cpu().numpy()
            ests[q] = est
            ref = sums[q]
            present = db(np.mean(ref ** 2)) > SILENT_DB
            row = {"track": t["id"], "dataset": t["dataset"], "query": q, "present": bool(present)}
            if present:
                row["sdr"] = sdr(est, ref)
                row["sdr_mix"] = sdr(mix, ref)
                row["sdri"] = row["sdr"] - row["sdr_mix"]
                row.update(diagnostics(est, sums, q, other_split=(other_sus, other_rest) if has_sub else None))
            else:
                row["leak_db"] = float(db(np.sum(est ** 2)) - db(np.sum(mix ** 2)))
            rows.append(row)
        if not blocked and len(ests) == 4 and db(np.mean(sums["other"] ** 2)) > SILENT_DB:
            other = mix - sum(ests.values())
            rows.append({"track": t["id"], "dataset": t["dataset"], "query": "other", "present": True,
                         "sdr": sdr(other, sums["other"]), "sdr_mix": sdr(mix, sums["other"]),
                         "sdri": sdr(other, sums["other"]) - sdr(mix, sums["other"])})
        el = time.time() - t0
        print(f"[{ti + 1}/{len(tracks)}] {t['id'][:50]:50s} {frames / 44100:5.0f}s  "
              f"(speed {audio_s / el:.0f}x real time over 4 queries)", flush=True)

    # summary: median over tracks (the convention in source-separation papers)
    summary = {}
    for ds in sorted({r["dataset"] for r in rows}):
        for q in config.USER_CLASSES:
            pr = [r for r in rows if r["dataset"] == ds and r["query"] == q and r["present"]]
            ab = [r for r in rows if r["dataset"] == ds and r["query"] == q and not r["present"]]
            if not pr and not ab:
                continue
            summary[f"{ds}/{q}"] = {
                "n_present": len(pr),
                "sdr_median": st.median(r["sdr"] for r in pr) if pr else None,
                "sdri_median": st.median(r["sdri"] for r in pr) if pr else None,
                "n_absent": len(ab),
                "leak_db_median": st.median(r["leak_db"] for r in ab) if ab else None,
            }
            for k in ("gain_median_db", "gain_spread_db", "dip_frac", "drums_leak_db", "partner_leak_db",
                      "sustained_leak_db"):
                v = [r[k] for r in pr if k in r]
                summary[f"{ds}/{q}"][k] = st.median(v) if v else None
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    json.dump({"ckpt": a.ckpt, "split": a.split, "window": a.window, "overlap": a.overlap,
               "summary": summary, "rows": rows}, open(a.out, "w"), indent=1)
    f = lambda v: "   -  " if v is None else f"{v:6.2f}"  # noqa: E731
    print(f"\n{'dataset/stem':22s} {'n':>3s} {'SDR':>6s} {'SDRi':>6s} | {'n_abs':>5s} {'leak':>6s} | "
          f"{'gain':>6s} {'spread':>6s} {'dips%':>6s} {'drumlk':>6s} {'pairlk':>6s} {'sustlk':>6s}")
    for k, v in summary.items():
        dip = None if v.get("dip_frac") is None else 100 * v["dip_frac"]
        print(f"{k:22s} {v['n_present']:3d} {f(v['sdr_median'])} {f(v['sdri_median'])} | "
              f"{v['n_absent']:5d} {f(v['leak_db_median'])} | {f(v.get('gain_median_db'))} "
              f"{f(v.get('gain_spread_db'))} {f(dip)} {f(v.get('drums_leak_db'))} "
              f"{f(v.get('partner_leak_db'))} {f(v.get('sustained_leak_db'))}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
