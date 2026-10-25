"""Separate a song into vocals / drums / guitar / piano / other with a checkpoint.

python scripts/separate.py --ckpt checkpoints/run1/best.pt --input song.wav --out sep_out/
python scripts/separate.py ... --target guitar        # one stem + its residual
"""
import argparse
import sys
from pathlib import Path

import soundfile as sf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sep import config  # noqa: E402
from sep.inference import load_model, separate, separate_all  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--input", required=True)
ap.add_argument("--out", default="sep_out")
ap.add_argument("--target", choices=config.QUERY_CLASSES)
ap.add_argument("--window", type=float, default=4.0)
ap.add_argument("--overlap", type=float, default=0.5)
ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
a = ap.parse_args()

audio, sr = sf.read(a.input, dtype="float32", always_2d=True)
if sr != config.SAMPLE_RATE:
    raise SystemExit(f"{a.input} is {sr} Hz; convert to {config.SAMPLE_RATE} Hz first")
mix = torch.from_numpy(audio.T.copy())
if mix.shape[0] == 1:
    mix = mix.repeat(2, 1)
mix = mix[:2]

model = load_model(a.ckpt, a.device)
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
stem_name = Path(a.input).stem
kw = dict(window_s=a.window, overlap=a.overlap)
if a.target:
    s = separate(model, mix, a.target, **kw).cpu()
    res = {a.target: s, "residual": mix - s}
else:
    res = {k: v.cpu() for k, v in separate_all(model, mix, **kw).items()}
for k, v in res.items():
    p = out / f"{stem_name}_{k}.wav"
    sf.write(p, v.T.numpy(), config.SAMPLE_RATE)
    print(p)
