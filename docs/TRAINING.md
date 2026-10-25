# Training Flora

The retained model is SCNet + FiLM. Older architecture experiments and lab notebooks are not part of this repository.

## Data

Obtain MoisesDB, MUSDB18-HQ, and Slakh2100 separately under their dataset terms. Keep test songs out of training. Audio paths in the generated manifest are machine-specific, so manifests are ignored by Git.

```bash
python scripts/build_manifest.py \
  --moisesdb /path/to/moisesdb/moisesdb_v0.1 \
  --musdb /path/to/musdb18hq \
  --slakh /path/to/slakh2100_flac_redux \
  --out manifest_all.json
```

Datasets may be omitted from indexing, but the recipes below assume all three. Source audio must be 44.1 kHz. Training generates four-second mixtures on the fly with varied gains and an instrument query. Absent-target examples teach silence. Hard examples emphasize confusable guitar and piano parts. Losses compare waveforms and spectrograms.

## Stage 1: train from scratch

```bash
python train.py --manifest manifest_all.json --out checkpoints/scnet1 \
  --arch scnet --scnet-width 128 --layers 6 --device cuda \
  --stft-weight 2 --under-weight 1 --lr 5e-4 --warmup 2000 \
  --steps 40000 --p-hard 0.5 \
  --dataset-weights moisesdb:0.75,slakh:0.15,musdb:0.1 \
  --val-dataset-weights moisesdb:1 \
  --query-weights vocals:1,drums:1,guitar:1.5,piano:1.5 \
  --best-on guitar,piano
```

## Stage 2: fine-tune on hard examples

```bash
python train.py --manifest manifest_all.json --out checkpoints/scnet2 \
  --arch scnet --scnet-width 128 --layers 6 --device cuda \
  --init-from checkpoints/scnet1/best.pt \
  --stft-weight 2 --under-weight 1 --lr 1.5e-4 --warmup 500 \
  --steps 20000 --absent-ramp 0,1 --p-hard 0.5 --p-hard-coh 0.7 \
  --dataset-weights moisesdb:0.75,slakh:0.15,musdb:0.1 \
  --val-dataset-weights moisesdb:1 \
  --query-weights vocals:1,drums:1,guitar:1.5,piano:1.5 \
  --hard-dataset-weights moisesdb:0.5,slakh:0.5 \
  --val-hard-items 256 --best-on guitar,piano,hard/guitar,hard/piano
```

These recipes expand the final-project README settings; training is stochastic and results depend on data and environment. Adjust batch size and workers to available resources. Re-running an output directory resumes `last.pt`; choose a new directory for a fresh experiment. `best.pt` is selected using the validation metrics requested by `--best-on`.

## Evaluate

```bash
python scripts/evaluate.py \
  --ckpt checkpoints/scnet2/best.pt \
  --manifest manifest_all.json --split test --out eval/local_test.json
```

Do not overwrite the recorded final results when experimenting. Use `--help` on `scripts/latency_bench.py` and `scripts/stream_loadtest.py` for compute and concurrent-listener benchmarks.
