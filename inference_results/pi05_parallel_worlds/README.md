# π0.5 × DreamDojo parallel worlds

This folder is a self-contained, reproducible result from the real model
pipeline:

1. Replay one observation from a recorded Jokeru episode.
2. Sample three 12-step action chunks from upstream `pi05_base`, keeping the
   observation and language prompt fixed and changing only the flow-matching
   noise seed.
3. Convert each 30D absolute action chunk to DreamDojo's 384D grouped-delta
   action condition.
4. Generate one 13-frame, 640 × 480 future per branch with the Jokeru
   DreamDojo 2B EMA checkpoint at iteration 3,000.

The current mock loop deliberately steals the next observation from the
recorded video. Generated frames are visualization rollouts and are not fed
back as VLA observations.

## Artifacts

- [`observation_replay.mp4`](observation_replay.mp4): recorded observation
  stream at the 7 Hz visualization cadence.
- `universe_XX_pi05_actions.npy`: π0.5 absolute actions, shape `[12, 30]`.
- `universe_XX_dreamdojo_actions.npy`: DreamDojo conditions, shape `[12, 384]`.
- `universe_XX.mp4`: generated 13-frame future.
- [`manifest.json`](manifest.json): prompts, seeds, timings, provenance, and
  pairwise output differences.

| Pair | Full-video pixel PSNR | Last-frame MAE |
|---|---:|---:|
| A–B | 18.444 dB | 0.04876 |
| A–C | 18.302 dB | 0.05213 |
| B–C | 20.018 dB | 0.03719 |

These branches use `pi05_base`, not a converged Jokeru LoRA checkpoint. A
one-step LoRA smoke run completed with loss `0.8082`; full three-epoch training
is configured but was not represented as complete.
