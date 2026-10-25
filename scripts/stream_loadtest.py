"""Load test for the streaming server: N simulated listeners, each streaming a real song in real
time (20 ms pieces, paced by the clock) and asking for a different instrument.

    python scripts/stream_loadtest.py --url ws://127.0.0.1:8765/ws --listeners 1,2,4,8 --seconds 30

Per listener and hop we record the delay = (time the hop arrived) - (time its first input sample
was sent), and check that `seq` has no gaps (no lost audio). Real time is sustained if the delay
stays flat (does not grow during the run) and the server backlog stays below one hop.
"""
import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import aiohttp
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep import config  # noqa: E402

HEADER = struct.Struct("<IqII")


def load_song(manifest, seconds):
    m = json.loads(Path(manifest).read_text())
    t = next(t for t in m["tracks"] if t["split"] == "test" and t["dataset"] == "moisesdb"
             and t["frames"] > (seconds + 60) * config.SAMPLE_RATE)
    n = int(seconds * config.SAMPLE_RATE)
    mix = np.zeros((2, n), np.float32)
    for s in t["stems"]:
        for p in s["paths"]:
            a, _ = sf.read(p, start=60 * config.SAMPLE_RATE, frames=n, dtype="float32", always_2d=True)
            mix[:, :a.shape[0]] += a.T[:2]
    return t["id"], mix


async def listener(url, mix, instrument, seconds, piece=882):
    delays, gaps, stats, sent_log = [], 0, [], []
    expect = 0
    async with aiohttp.ClientSession() as s:
        async with s.ws_connect(url, max_msg_size=16 * 2**20) as ws:
            await ws.send_json({"type": "select", "instrument": instrument})
            done = asyncio.Event()

            async def recv():
                nonlocal gaps, expect
                async for msg in ws:
                    now = time.perf_counter()
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        seq, start, n, _ = HEADER.unpack_from(msg.data)
                        if seq != expect:
                            gaps += seq - expect
                        expect = seq + 1
                        i = np.searchsorted([f for f, _ in sent_log], start, side="right") - 1
                        if i >= 0:
                            delays.append((sent_log[i][1] - t0, now - sent_log[i][1]))
                        if start + n >= mix.shape[1] - 2 * config.SAMPLE_RATE:
                            done.set()
                    elif msg.type == aiohttp.WSMsgType.TEXT:
                        m = json.loads(msg.data)
                        if m.get("type") == "stats":
                            stats.append(m)

            t0 = time.perf_counter()
            rtask = asyncio.create_task(recv())
            sent = 0
            total = mix.shape[1]
            while sent < total:
                due = min(total, int((time.perf_counter() - t0) * config.SAMPLE_RATE))
                if due > sent:
                    sent_log.append((sent, time.perf_counter()))
                    await ws.send_bytes(np.ascontiguousarray(mix[:, sent:due].T, dtype="<f4").tobytes())
                    sent = due
                await asyncio.sleep(piece / config.SAMPLE_RATE)
            try:
                await asyncio.wait_for(done.wait(), timeout=10)
            except asyncio.TimeoutError:
                pass
            await ws.close()
            rtask.cancel()
    return delays, gaps, stats


async def run(a):
    song, mix = load_song(a.manifest, a.seconds)
    print(f"song {song}, {a.seconds}s", flush=True)
    instr = [x for x in a.instruments.split(",") if x]
    results = []
    for n in [int(x) for x in a.listeners.split(",")]:
        res = await asyncio.gather(*(listener(a.url, mix, instr[i % len(instr)], a.seconds) for i in range(n)))
        d = np.array([x for r in res for _, x in r[0]])
        first = np.array([x for r in res for t, x in r[0] if t < 10])
        last = np.array([x for r in res for t, x in r[0] if t > a.seconds - 10])
        backlog = max((s["backlog_s"] for r in res for s in r[2]), default=None)
        row = {"listeners": n, "hops": int(d.size), "lost_pieces": int(sum(r[1] for r in res)),
               "delay_p50_s": round(float(np.median(d)), 3), "delay_p95_s": round(float(np.percentile(d, 95)), 3),
               "delay_max_s": round(float(d.max()), 3),
               "delay_first10s_p50": round(float(np.median(first)), 3) if first.size else None,
               "delay_last10s_p50": round(float(np.median(last)), 3) if last.size else None,
               "max_backlog_s": backlog}
        row["keeps_up"] = bool(row["lost_pieces"] == 0 and row["delay_last10s_p50"] is not None
                               and row["delay_last10s_p50"] - row["delay_first10s_p50"] < 0.25)
        print(row, flush=True)
        results.append(row)
        await asyncio.sleep(2)
    if a.out:
        Path(a.out).parent.mkdir(parents=True, exist_ok=True)
        json.dump({"song": song, "seconds": a.seconds, "rows": results}, open(a.out, "w"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="ws://127.0.0.1:8765/ws")
    ap.add_argument("--manifest", default="manifest_eval.json")
    ap.add_argument("--listeners", default="1,2,4,8")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--instruments", default="vocals,drums,piano,other",
                    help="listener i asks for instruments[i %% len]; must be offered by the server")
    ap.add_argument("--out", default="")
    asyncio.run(run(ap.parse_args()))


if __name__ == "__main__":
    main()
