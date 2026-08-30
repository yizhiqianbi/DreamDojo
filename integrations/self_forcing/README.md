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

Run a one-sample teacher-cache smoke test on a free GPU:

```bash
CUDA_VISIBLE_DEVICES=1 NPROC=1 TEACHER_END=1 \
  bash scripts/run_jokeru_self_forcing.sh teacher-cache
```

Full training expects eight free GPUs. Set `WARMUP_CHECKPOINT` to the actual warmup iteration before starting Self-Forcing if the output layout differs from the default:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
  bash scripts/run_jokeru_self_forcing.sh warmup

WARMUP_CHECKPOINT=/absolute/path/to/iter_000020000 \
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 NPROC=8 \
  bash scripts/run_jokeru_self_forcing.sh self-forcing
```

The NVIDIA repository does not currently include a downloadable Jokeru action-conditioned Self-Forcing student checkpoint. Until the two student training stages finish, existing parallel-world videos remain 35-step teacher outputs and must not be reported as Self-Forcing inference.
