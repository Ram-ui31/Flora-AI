"""Real-time separation server (WebSocket) + browser client.

    python scripts/stream_server.py --ckpt checkpoints/run6/best.pt --window 1.0      # CPU (default)
    python scripts/stream_server.py --ckpt ... --device cuda                          # GPU

then open http://localhost:8765 (from a laptop: ssh -L 8765:localhost:8765 university-gpu).

Protocol (one WebSocket per listener, /ws)
------------------------------------------
client -> server  text   {"type": "select", "instrument": "guitar"}      any time (cross-faded)
client -> server  binary float32 little-endian, interleaved stereo, 44.1 kHz, any piece size
server -> client  text   {"type": "config", ...} once, then {"type": "stats", ...} about once a second
server -> client  binary header <I seq><q start><I n><I instrument index> (20 bytes), then
                         stem (n*2 float32 interleaved), then rest = input - stem (n*2 float32)

`start` is the index of the hop's first sample in the listener's own input stream, so the client
can measure the exact delay (time the sample was sent -> time it came back) and detect gaps
(`seq` is consecutive).

Concurrency: the event loop only moves bytes. All model calls go to ONE InferenceWorker that
batches windows from every listener (any mix of instruments) and runs them in a background
thread, so receiving/sending never waits for the model. Each listener's windows are processed in
order; if the model falls behind, that listener's backlog grows and is reported - audio is never
silently dropped.
"""
import argparse
import asyncio
import json
import struct
import sys
import time
from pathlib import Path

import numpy as np
import torch
from aiohttp import WSMsgType, web

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from sep import config  # noqa: E402
from sep.inference import load_model  # noqa: E402
from sep.stream.engine import InferenceWorker, StreamState, process_window  # noqa: E402

HEADER = struct.Struct("<IqII")


def pack_hop(h, instr):
    stem = np.ascontiguousarray(h.stem.T, dtype="<f4")           # (n, 2) interleaved
    rest = np.ascontiguousarray(h.rest.T, dtype="<f4")
    return HEADER.pack(h.seq, h.start, h.stem.shape[1], instr.index(h.instrument)) + stem.tobytes() + rest.tobytes()


async def ws_handler(request):
    app = request.app
    a = app["args"]
    ws = web.WebSocketResponse(max_msg_size=8 * 2**20, heartbeat=20)
    await ws.prepare(request)
    instr = app["instruments"]
    st = StreamState(a.window, a.overlap, a.instrument, allowed=instr)
    app["listeners"].add(st)
    await ws.send_json({"type": "config", "sample_rate": config.SAMPLE_RATE, "channels": 2,
                        "window_s": st.W / config.SAMPLE_RATE, "hop_s": st.H / config.SAMPLE_RATE,
                        "delay_s": [st.P / config.SAMPLE_RATE, st.W / config.SAMPLE_RATE],
                        "instruments": instr, "instrument": st.instrument, "device": app["device_name"],
                        "model": app["model_name"]})
    ready = asyncio.Event()
    stats = {"hops": 0, "compute_ms": []}

    async def pump():
        last = time.monotonic()
        while True:
            await ready.wait()
            ready.clear()
            while st.pending.shape[1] >= st.H:
                t0 = time.perf_counter()
                h = await process_window(st, app["worker"], app["other_from"])
                stats["compute_ms"].append(1000 * (time.perf_counter() - t0))
                if h is not None:
                    await ws.send_bytes(pack_hop(h, instr))
                    stats["hops"] += 1
            if time.monotonic() - last > 1.0:
                last = time.monotonic()
                c = stats["compute_ms"][-50:]
                await ws.send_json({"type": "stats", "backlog_s": round(st.backlog_s(), 3),
                                    "window_ms_p50": round(float(np.median(c)), 1) if c else None,
                                    "window_ms_max": round(float(np.max(c)), 1) if c else None,
                                    "listeners": len(app["listeners"]),
                                    "received_s": round(st.received / config.SAMPLE_RATE, 2),
                                    "instrument": st.instrument})

    task = asyncio.create_task(pump())
    try:
        async for msg in ws:
            if msg.type == WSMsgType.BINARY:
                pcm = np.frombuffer(msg.data, dtype="<f4")
                if pcm.size % 2:
                    continue
                st.feed(pcm.reshape(-1, 2).T)
                ready.set()
            elif msg.type == WSMsgType.TEXT:
                m = json.loads(msg.data)
                if m.get("type") == "select":
                    try:
                        st.select(m["instrument"])
                    except ValueError as e:
                        await ws.send_json({"type": "error", "message": str(e)})
            elif msg.type == WSMsgType.ERROR:
                break
    finally:
        task.cancel()
        app["listeners"].discard(st)
    return ws


async def index(request):
    return web.FileResponse(ROOT / "web" / "index.html")


async def demo_list(request):
    """Built-in demo songs: <demo-dir>/index.json = [{"file": "...wav", "title": "..."}]."""
    d = request.app["demo_dir"]
    f = d / "index.json"
    return web.json_response(json.loads(f.read_text()) if f.exists() else [])


async def demo_file(request):
    name = request.match_info["name"]
    d = request.app["demo_dir"]
    p = (d / name).resolve()
    if p.parent != d.resolve() or p.suffix.lower() not in (".wav", ".flac", ".mp3") or not p.exists():
        raise web.HTTPNotFound()
    return web.FileResponse(p)


async def on_startup(app):
    app["worker"].start()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="checkpoints/scnet2/best.pt")
    ap.add_argument("--instruments", default="vocals,drums,piano,other",
                    help="choices offered to listeners; 'other' = mix minus the other listed instruments")
    ap.add_argument("--demo-dir", default="demo")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8, help="torch CPU threads")
    ap.add_argument("--window", type=float, default=1.0, help="seconds; delay is between W-H and W")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--instrument", default="vocals", help="starting instrument")
    ap.add_argument("--max-batch", type=int, default=16)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8765)
    a = ap.parse_args()
    torch.set_num_threads(a.threads)
    model = load_model(a.ckpt, a.device)
    app = web.Application()
    app["args"] = a
    app["worker"] = InferenceWorker(model, a.device, a.max_batch)
    app["listeners"] = set()
    instr = [x for x in a.instruments.split(",") if x]
    bad = [x for x in instr if x not in config.USER_CLASSES]
    if bad:
        raise SystemExit(f"unknown instruments {bad}")
    app["instruments"] = instr
    app["other_from"] = tuple(q for q in instr if q != "other") if "other" in instr else config.QUERY_CLASSES
    app["model_name"] = Path(a.ckpt).parent.name
    app["demo_dir"] = (ROOT / a.demo_dir) if not Path(a.demo_dir).is_absolute() else Path(a.demo_dir)
    app["device_name"] = torch.cuda.get_device_name(0) if a.device.startswith("cuda") else f"cpu x{a.threads}"
    app.router.add_get("/", index)
    app.router.add_get("/ws", ws_handler)
    app.router.add_get("/demo", demo_list)
    app.router.add_get("/demo/{name}", demo_file)
    app.on_startup.append(on_startup)
    print(f"model {a.ckpt} on {app['device_name']}; instruments {instr} (other = mix - {list(app['other_from'])}); "
          f"window {a.window}s overlap {a.overlap}; "
          f"http://{a.host}:{a.port}", flush=True)
    web.run_app(app, host=a.host, port=a.port, print=None)


if __name__ == "__main__":
    main()
