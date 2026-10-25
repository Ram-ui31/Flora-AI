"""How fast is one streaming step? For each window length W (hop = W/2), time one forward pass
on a single window, the way the streaming server will run it (one listener, one instrument).

    CUDA_VISIBLE_DEVICES= python scripts/latency_bench.py --ckpt checkpoints/run6/best.pt   # CPU
    CUDA_VISIBLE_DEVICES=2 python scripts/latency_bench.py --ckpt ...                       # GPU

Real-time needs: compute time per window < hop (W/2), otherwise the queue grows.
Delay a listener hears ~= W (wait for the window to fill, worst case) + compute time.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep import config  # noqa: E402
from sep.inference import load_model  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--windows", default="4,2,1,0.5")
    ap.add_argument("--threads", type=int, default=0, help="torch CPU threads (0 = default)")
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(a.ckpt, dev).eval()
    res = {"device": dev if dev == "cpu" else torch.cuda.get_device_name(0),
           "threads": torch.get_num_threads(), "rows": []}
    for w in [float(x) for x in a.windows.split(",")]:
        n = int(w * config.SAMPLE_RATE)
        x = torch.randn(1, 2, n, device=dev) * 0.1
        q = torch.tensor([2], device=dev)
        with torch.inference_mode():
            for _ in range(2):                       # warm-up
                model(x, q)
            if dev == "cuda":
                torch.cuda.synchronize()
            t = []
            for _ in range(a.reps):
                t0 = time.perf_counter()
                model(x, q)
                if dev == "cuda":
                    torch.cuda.synchronize()
                t.append(time.perf_counter() - t0)
        t.sort()
        med = t[len(t) // 2]
        hop = w / 2
        row = dict(window_s=w, hop_s=hop, compute_ms=round(1000 * med, 1),
                   load=round(med / hop, 3),                      # < 1 = keeps up in real time
                   delay_s=round(w + med, 3))
        if dev == "cuda":
            row["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 2**20)
        res["rows"].append(row)
        print(row, flush=True)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump(res, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
