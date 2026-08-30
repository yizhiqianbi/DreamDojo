# Jokeru Self-Forcing

This integration adapts DreamDojo's official Self-Forcing path to the local Jokeru action-conditioned teacher. It preserves the native `384D` action embedding and `12` actions per causal chunk.

The pipeline has three training stages:

1. Cache selected states from the completed 35-step Jokeru teacher trajectories.
2. Warm up a causal student against those cached states.
3. Train the student with Self-Forcing DMD on 49-frame Jokeru windows, then run 4-step streaming inference.

Check local readiness without touching a GPU:

```bash
bash scripts/run_jokeru_self_forcing.sh status
```

Download the local Wan2.1 VAE and CR1 empty-prompt embedding required by the
training stages (the latter is mirrored publicly because the upstream Cosmos
repository is gated):

```bash
python scripts/download_jokeru_assets.py --skip-datasets
python scripts/validate_jokeru_assets.py
```

Run a one-sample teacher-cache smoke test on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=1 NPROC=1 TEACHER_END=1 \
  bash scripts/run_jokeru_self_forcing.sh teacher-cache
```

Full training is designed for eight GPUs. The launcher discovers the latest
warmup checkpoint under `WARMUP_OUTPUT`; `WARMUP_CHECKPOINT` can still select a
specific iteration explicitly:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
  bash scripts/run_jokeru_self_forcing.sh warmup

WARMUP_CHECKPOINT=/absolute/path/to/iter_000020000 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
  bash scripts/run_jokeru_self_forcing.sh self-forcing
```

## Verified co-resident smoke run

On 2026-08-30 the complete path was exercised on eight H200 GPUs while eight
existing teacher-inference jobs remained resident:

- Teacher cache: 8 complete trajectories, about 20 seconds per sample/GPU.
- Causal warmup, batch 1/GPU: step 1 took 38.31 seconds, step 2 took 13.68 seconds;
  loss changed from 0.0590 to 0.0189 and both checkpoints were saved.
- Self-Forcing, batch 1/GPU: one step took 123.22 seconds and produced a complete
  DCP student checkpoint.
- Peak memory attributable to warmup was 14.16 GB/GPU; Self-Forcing peaked at
  34.54 GB/GPU. With the resident inference jobs, the tightest observed margin
  was about 5.2 GB, so production training should use free GPUs even though the
  smoke run did not OOM.

The one-step student checkpoint only proves the pipeline and is not a
quality-trained model.

The NVIDIA repository does not currently include a downloadable Jokeru action-conditioned Self-Forcing student checkpoint. Until the two student training stages finish, existing parallel-world videos remain 35-step teacher outputs and must not be reported as Self-Forcing inference.
