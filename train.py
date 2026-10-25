"""Train the conditioned separator.

python train.py --manifest manifest.json --out checkpoints/scnet1
Re-running the same command resumes from <out>/last.pt automatically.

Outputs in --out:
    last.pt            every --ckpt-every steps (resume point)
    best.pt            best validation mean-SDR so far
    step_XXXXXXX.pt    snapshot every --keep-every steps
    metrics.jsonl      one JSON line per log / validation event
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sep import config
from sep.data import ConditionedMixDataset, MixerConfig, load_manifest
from sep.losses import conditioned_loss, rel_energy_db, sdr
from sep.models import build_model, count_params


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", required=True)
    p.add_argument("--out", default="checkpoints/scnet1")
    # data
    p.add_argument("--segment", type=float, default=4.0)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--dataset-weights", type=str, default=None,
                   help='JSON, e.g. \'{"moisesdb":0.5,"musdb":0.5}\'')
    p.add_argument("--p-absent", type=float, default=0.15)
    p.add_argument("--p-hard", type=float, default=0.0,
                   help="piano/guitar: prob. of a hard mix (partner + sustained 'other' part)")
    p.add_argument("--query-weights", type=str, default=None,
                   help='JSON, e.g. \'{"vocals":1,"drums":1,"guitar":1.25,"piano":1.5}\'')
    p.add_argument("--val-dataset-weights", type=str, default=None,
                   help="JSON; defaults to --dataset-weights. Keep validation comparable across runs.")
    p.add_argument("--init-from", type=str, default=None,
                   help="start a NEW run from these model weights (fresh optimiser, step 0)")
    p.add_argument("--absent-weight", type=float, default=10.0,
                   help="weight of the linear leak penalty on absent queries (after ramp)")
    p.add_argument("--absent-ramp", type=str, default="4000,20000",
                   help="absent penalty is 0 before the first step and full after the second")
    # optimisation
    p.add_argument("--steps", type=int, default=100_000)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--min-lr", type=float, default=1e-5)
    p.add_argument("--warmup", type=int, default=2000)
    p.add_argument("--clip", type=float, default=5.0)
    p.add_argument("--wd", type=float, default=0.01, help="AdamW weight decay (bounds weight growth)")
    p.add_argument("--reset-best", action="store_true", help="on resume, forget the stored best score")
    p.add_argument("--best-on", type=str, default="",
                   help="comma list of classes whose mean val SDR picks best.pt (default: all 4)")
    p.add_argument("--max-bad-streak", type=int, default=200, help="abort after this many non-finite batches in a row")
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    # model
    p.add_argument("--arch", choices=["scnet"], default="scnet")
    p.add_argument("--scnet-width", type=int, default=128, help="scnet channels of the last level (128 base, 256 large)")
    p.add_argument("--layers", type=int, default=6, help="SCNet dual-path layers")
    p.add_argument("--stft-weight", type=float, default=0.0, help="weight of multi-res STFT loss (present targets)")
    p.add_argument("--under-weight", type=float, default=0.0,
                   help="extra STFT log-magnitude cost for bins where the estimate is too QUIET (0 = off)")
    # bookkeeping
    p.add_argument("--log-every", type=int, default=100)
    p.add_argument("--val-every", type=int, default=2000)
    p.add_argument("--val-items", type=int, default=512)
    p.add_argument("--val-hard-items", type=int, default=0,
                   help="extra validation set: guitar/piano queries on same-song excerpts where both play "
                        "(MoisesDB valid songs); logged as hard/sdr/<class>. 0 = off")
    p.add_argument("--p-hard-coh", type=float, default=0.0,
                   help="share of hard guitar/piano examples taken from one song where both play")
    p.add_argument("--hard-dataset-weights", type=str, default=None,
                   help="dataset weights for those same-song hard examples, e.g. moisesdb:0.5,slakh:0.5")
    p.add_argument("--ckpt-every", type=int, default=1000)
    p.add_argument("--keep-every", type=int, default=20_000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="auto")
    return p.parse_args()


def parse_weights(text):
    """'{"a":1,"b":2}' (JSON) or 'a:1,b:2' (shell-safe: no braces/quotes to mangle over ssh)."""
    if text is None:
        return None
    text = text.strip()
    if text.startswith("{"):
        return json.loads(text)
    return {k.strip(): float(v) for k, v in (kv.split(":") for kv in text.split(",") if kv.strip())}


def lr_at(step, a):
    if step < a.warmup:
        return a.lr * (step + 1) / a.warmup
    t = min(1.0, (step - a.warmup) / max(1, a.steps - a.warmup))
    return a.min_lr + 0.5 * (a.lr - a.min_lr) * (1 + math.cos(math.pi * t))


def absent_w(step, a):
    s0, s1 = (int(x) for x in a.absent_ramp.split(","))
    if step < s0:
        return 0.0
    return a.absent_weight * min(1.0, (step - s0) / max(1, s1 - s0))


def save_atomic(obj, path: Path):
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


@torch.no_grad()
def validate(model, loader, device, use_amp):
    model.eval()
    sdrs = {q: [] for q in config.QUERY_CLASSES}
    absent = {q: [] for q in config.QUERY_CLASSES}
    nonfinite = 0
    for b in loader:
        mix, tgt = b["mix"].to(device), b["target"].to(device)
        q, present = b["query"].to(device), b["present"].to(device)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            est = model(mix, q)
        s, e = sdr(est, tgt), rel_energy_db(est, mix)
        ok = torch.isfinite(est).flatten(1).all(1)
        nonfinite += int((~ok).sum())
        for i in range(len(q)):
            if not ok[i]:
                continue
            name = config.QUERY_CLASSES[int(q[i])]
            (sdrs if present[i] else absent)[name].append(float(s[i] if present[i] else e[i]))
    model.train()
    mean = lambda v: sum(v) / len(v) if v else float("nan")  # noqa: E731
    res = {f"sdr/{k}": mean(v) for k, v in sdrs.items()}
    res.update({f"absent_db/{k}": mean(v) for k, v in absent.items()})
    vals = [v for k, v in res.items() if k.startswith("sdr/") and not math.isnan(v)]
    res["sdr/mean"] = sum(vals) / len(vals) if vals else float("nan")
    res["nonfinite"] = nonfinite
    return res


def main():
    a = get_args()
    torch.manual_seed(a.seed)
    torch.backends.cudnn.benchmark = True
    device = torch.device(("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device)
    use_amp = a.amp and device.type == "cuda"
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    mlog = open(out / "metrics.jsonl", "a")

    def log(d):
        d["time"] = time.time()
        mlog.write(json.dumps(d) + "\n")
        mlog.flush()

    manifest = load_manifest(a.manifest)
    mk = dict(segment_seconds=a.segment, p_absent=a.p_absent)
    if a.dataset_weights:
        mk["dataset_weights"] = parse_weights(a.dataset_weights)
    train_kw = dict(mk, p_hard=a.p_hard, p_hard_coh=a.p_hard_coh)
    if a.hard_dataset_weights:
        train_kw["hard_dataset_weights"] = parse_weights(a.hard_dataset_weights)
    if a.query_weights:
        train_kw["query_weights"] = parse_weights(a.query_weights)
    val_kw = dict(mk)
    if a.val_dataset_weights:
        val_kw["dataset_weights"] = parse_weights(a.val_dataset_weights)
    train_ds = ConditionedMixDataset(manifest, MixerConfig(split="train", epoch_size=10**9, **train_kw))
    val_ds = ConditionedMixDataset(manifest, MixerConfig(
        split="valid", deterministic=True, epoch_size=a.val_items, seed=1234, **val_kw))
    val_hard_dl = None
    if a.val_hard_items:
        hk = dict(val_kw, p_hard=1.0, p_hard_coh=1.0, query_weights={"guitar": 1, "piano": 1},
                  dataset_weights={"moisesdb": 1.0}, hard_dataset_weights={"moisesdb": 1.0})
        val_hard_ds = ConditionedMixDataset(manifest, MixerConfig(
            split="valid", deterministic=True, epoch_size=a.val_hard_items, seed=4321, **hk))
        print(f"hard validation: {len(val_hard_ds.coh_both.get('moisesdb', []))} MoisesDB valid songs "
              f"with guitar+piano, {a.val_hard_items} items", flush=True)
    dl_kw = dict(num_workers=a.workers, pin_memory=device.type == "cuda",
                 persistent_workers=a.workers > 0)
    train_dl = DataLoader(train_ds, batch_size=a.batch, drop_last=True, **dl_kw)
    if a.val_hard_items:
        val_hard_dl = DataLoader(val_hard_ds, batch_size=a.batch, **dl_kw)
    val_dl = DataLoader(val_ds, batch_size=a.batch, **dl_kw)

    model = build_model(vars(a), n_queries=len(config.QUERY_CLASSES)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=a.lr, weight_decay=a.wd)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    step, best = 0, -float("inf")
    best_cls = [c for c in a.best_on.split(",") if c] or None

    def score(res):
        if not best_cls:
            return res["sdr/mean"]
        # "guitar" -> sdr/guitar ; "hard/guitar" -> hard/sdr/guitar
        key = lambda c: f"hard/sdr/{c[5:]}" if c.startswith("hard/") else f"sdr/{c}"  # noqa: E731
        return sum(res[key(c)] for c in best_cls) / len(best_cls)

    last = out / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        scaler.load_state_dict(ck["scaler"])
        step, best = ck["step"], ck["best"]
        for g in opt.param_groups:          # load_state_dict restores old hyper-params
            g["weight_decay"] = a.wd
        if a.reset_best:
            best = -float("inf")
        print(f"resumed from {last} at step {step} (best sdr {best:.2f})", flush=True)
        torch.manual_seed(a.seed + step)   # fresh data stream after resume, not a replay
    else:
        if a.init_from:
            init = torch.load(a.init_from, map_location=device, weights_only=False)
            model.load_state_dict(init["model"])
            print(f"initialised weights from {a.init_from} (its step {init.get('step')})", flush=True)
        print(f"new run: {count_params(model):.2f}M params, device={device}, amp={use_amp}", flush=True)
    print(json.dumps(vars(a)), flush=True)

    def ckpt():
        return {"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "step": step, "best": best, "args": vars(a),
                "query_classes": config.QUERY_CLASSES}

    model.train()
    t0, run_loss, run_n = time.time(), 0.0, 0
    run_parts = {"loss_present": [], "stft_present": [], "leak_absent": []}
    bad_streak, bad_total = 0, 0
    it = iter(train_dl)
    while step < a.steps:
        b = next(it)
        mix = b["mix"].to(device, non_blocking=True)
        tgt = b["target"].to(device, non_blocking=True)
        q = b["query"].to(device, non_blocking=True)
        present = b["present"].to(device, non_blocking=True)

        for g in opt.param_groups:
            g["lr"] = lr_at(step, a)
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            est = model(mix, q)
        per_item, parts = conditioned_loss(est, tgt, mix, present, absent_w(step, a), a.stft_weight, a.under_weight)
        loss = per_item.mean()
        if not torch.isfinite(loss):
            bad_streak += 1
            bad_total += 1
            print(f"step {step}: non-finite loss, skipping batch ({bad_streak} in a row)", flush=True)
            opt.zero_grad(set_to_none=True)
            if bad_streak >= a.max_bad_streak:
                raise SystemExit(f"{bad_streak} non-finite batches in a row at step {step}; aborting")
            continue
        bad_streak = 0
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        gnorm = torch.nn.utils.clip_grad_norm_(model.parameters(), a.clip)
        scaler.step(opt)
        scaler.update()
        step += 1
        run_loss += loss.item()
        run_n += 1
        for k, v in parts.items():
            if not math.isnan(v):
                run_parts[k].append(v)

        if step % a.log_every == 0:
            dt = time.time() - t0
            mem = torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0
            msg = {"step": step, "loss": run_loss / run_n,
                   "present_snr_loss": sum(run_parts["loss_present"]) / max(1, len(run_parts["loss_present"])),
                   "stft_loss": sum(run_parts["stft_present"]) / max(1, len(run_parts["stft_present"])),
                   "absent_leak": sum(run_parts["leak_absent"]) / max(1, len(run_parts["leak_absent"])),
                   "absent_w": absent_w(step, a), "lr": lr_at(step, a),
                   "gnorm": float(gnorm), "it_s": run_n / dt, "gpu_gb": mem, "skipped_total": bad_total}
            print(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                           for k, v in msg.items()), flush=True)
            log({"type": "train", **msg})
            t0, run_loss, run_n = time.time(), 0.0, 0
            run_parts = {"loss_present": [], "stft_present": [], "leak_absent": []}

        if step % a.val_every == 0 or step == a.steps:
            res = validate(model, val_dl, device, use_amp)
            if val_hard_dl is not None:
                rh = validate(model, val_hard_dl, device, use_amp)
                res.update({f"hard/{k}": v for k, v in rh.items()
                            if k.split("/")[-1] in ("guitar", "piano") and not k.startswith("sdr/mean")})
                res["nonfinite"] += rh["nonfinite"]
            print("VAL step=%d " % step + " ".join(f"{k}={v:.2f}" for k, v in res.items()), flush=True)
            log({"type": "val", "step": step, **res})
            healthy = res["nonfinite"] == 0
            if not healthy:
                print(f"validation NOT healthy ({res['nonfinite']} non-finite outputs); best not updated", flush=True)
            elif score(res) > best:
                best = score(res)
                save_atomic(ckpt(), out / "best.pt")
                print(f"new best {best:.2f} dB -> {out / 'best.pt'}", flush=True)

        if step % a.ckpt_every == 0 or step == a.steps:
            save_atomic(ckpt(), last)
        if step % a.keep_every == 0:
            save_atomic(ckpt(), out / f"step_{step:07d}.pt")

    print("done", flush=True)


if __name__ == "__main__":
    main()
