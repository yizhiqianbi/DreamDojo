# π0.5 × DreamDojo — 48-step parallel worlds

This is the real 48-step extension of the parallel-world pipeline. For the
same recorded Jokeru observation and instruction, upstream `pi05_base` directly
samples three independent `[48, 30]` absolute-action plans. The plans are not
made by concatenating four separate VLA calls.

DreamDojo is kept on the 12-action window used during Jokeru post-training. Its
`[48, 384]` condition is rolled through four chained windows. The last frame of
each window seeds the next; duplicated boundary frames are removed, producing
one 49-frame, 640 × 480 video at 7.25 FPS per universe.

## Artifacts

- [`observation_replay.mp4`](observation_replay.mp4): 49 recorded frames for
  comparison, not a generated observation.
- `universe_XX_pi05_actions.npy`: one-shot π0.5 plans, shape `[48, 30]`.
- `universe_XX_dreamdojo_actions.npy`: bridged conditions, shape `[48, 384]`.
- `universe_XX.mp4`: four-window autoregressive DreamDojo result, 49 frames.
- [`manifest.json`](manifest.json): seeds, action timings, four window ranges,
  provenance, and pairwise video differences.

| Branch | Action RMS from A | π0.5 inference |
|---|---:|---:|
| A | 0.0000 | 15,336.5 ms cold/JIT |
| B | 0.0986 | 74.2 ms warm |
| C | 0.1094 | 81.5 ms warm |

| Pair | Full-video pixel PSNR | Last-frame MAE |
|---|---:|---:|
| A–B | 18.116 dB | 0.05635 |
| A–C | 18.149 dB | 0.04652 |
| B–C | 17.354 dB | 0.05202 |

All videos decode to exactly 49 frames. Motion continuity was checked at the
three chained boundaries; their frame-to-frame MAE lies inside the normal
rollout motion range. The branches use `pi05_base`, not a completed Jokeru
LoRA checkpoint.
