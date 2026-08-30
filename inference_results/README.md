# DreamDojo 2B × Jokeru inference results

This directory contains our reproducible evaluation artifacts for the Jokeru
post-training run. All results use the BF16 EMA checkpoint at iteration 3,000.
The model input is the first RGB frame plus a sequence of robot actions; the
output is an action-conditioned video continuation.

## π0.5 48-step parallel-world results

[`pi05_parallel_worlds_48`](pi05_parallel_worlds_48) contains one recorded
Jokeru observation, three 48-step action plans sampled in one inference each by
the real π0.5 base policy, and the three corresponding 49-frame DreamDojo
futures. DreamDojo chains four native 12-action windows per branch. The
branches use explicit Gaussian flow-matching noise seeds
`20260830`–`20260832`; they are samples from the policy, not hand-perturbed
actions.

| Branch | Action RMS from A | Warm/cold policy time | Video |
|---|---:|---:|---|
| A | 0.0000 | 15,336.5 ms (JIT cold start) | [video](pi05_parallel_worlds_48/universe_00.mp4) |
| B | 0.0986 | 74.2 ms | [video](pi05_parallel_worlds_48/universe_01.mp4) |
| C | 0.1094 | 81.5 ms | [video](pi05_parallel_worlds_48/universe_02.mp4) |

Pairwise full-video pixel PSNR is 17.35–18.15 dB, confirming that the same
observation and prompt lead to visibly different generated futures. These
artifacts use the upstream `pi05_base` weights before Jokeru fine-tuning. The
Jokeru LoRA path was independently validated with a successful one-step train
and checkpoint save; it must not be confused with a converged policy. See the
[full integration guide](../docs/PI05_PARALLEL_WORLDS.md) and the
[interactive viewer](https://dreamdojo-jokeru-lab.boingshaw.chatgpt.site).

The earlier 12-step baseline is preserved under
[`pi05_parallel_worlds`](pi05_parallel_worlds).

## Checkpoint and protocol

- Model: DreamDojo 2B, Jokeru post-training checkpoint `iter_000003000`
- Native window: 1 initial frame + 12 actions → 13 video frames
- Resolution: 480 × 640
- Short evaluation: one held-out window from each of 13 Jokeru datasets
- Long-horizon evaluation: 1 initial frame + 60 actions → 61 frames, generated
  autoregressively as five consecutive 12-action chunks
- Weight repository: [PencilHu/DreamDojo-Jokeru-2B-iter3000](https://huggingface.co/PencilHu/DreamDojo-Jokeru-2B-iter3000)

Each `*_merged.mp4` places ground truth on the left and the prediction on the
right. The raw prediction, ground truth, action tensor, and per-sample metrics
are stored beside it.

## Short-horizon results

Aggregate over all 13 datasets: **PSNR 27.294**, **SSIM 0.943**, **LPIPS
0.086**.

| # | Dataset/task | PSNR | SSIM | LPIPS | Comparison |
|---:|---|---:|---:|---:|---|
| 00 | arrange patch | 28.286 | 0.958 | 0.062 | [video](iter_000003000/short_13_datasets/0000_merged.mp4) |
| 01 | arrange patch, annotated | 28.286 | 0.958 | 0.062 | [video](iter_000003000/short_13_datasets/0001_merged.mp4) |
| 02 | grip bottle cap | 27.279 | 0.933 | 0.137 | [video](iter_000003000/short_13_datasets/0002_merged.mp4) |
| 03 | arrange set 2 | 25.061 | 0.918 | 0.092 | [video](iter_000003000/short_13_datasets/0003_merged.mp4) |
| 04 | arrange set 2, annotated | 25.061 | 0.918 | 0.092 | [video](iter_000003000/short_13_datasets/0004_merged.mp4) |
| 05 | arrange set 3, annotated | 27.859 | 0.951 | 0.075 | [video](iter_000003000/short_13_datasets/0005_merged.mp4) |
| 06 | continuous shelf organizing | 24.127 | 0.906 | 0.102 | [video](iter_000003000/short_13_datasets/0006_merged.mp4) |
| 07 | arrange base | 27.608 | 0.950 | 0.074 | [video](iter_000003000/short_13_datasets/0007_merged.mp4) |
| 08 | arrange base, annotated | 27.608 | 0.950 | 0.074 | [video](iter_000003000/short_13_datasets/0008_merged.mp4) |
| 09 | arrange set 3 | 27.859 | 0.951 | 0.075 | [video](iter_000003000/short_13_datasets/0009_merged.mp4) |
| 10 | pick up black bottle | 34.978 | 0.976 | 0.117 | [video](iter_000003000/short_13_datasets/0010_merged.mp4) |
| 11 | take wrong item, right arm | 24.760 | 0.947 | 0.077 | [video](iter_000003000/short_13_datasets/0011_merged.mp4) |
| 12 | pick purple box and place in middle | 26.052 | 0.944 | 0.073 | [video](iter_000003000/short_13_datasets/0012_merged.mp4) |

## Action-sensitivity controls

Two inputs were regenerated after replacing every action with zero. Aggregate
quality drops to **PSNR 19.858**, **SSIM 0.852**, **LPIPS 0.141**, compared with
**PSNR 24.594**, **SSIM 0.912**, **LPIPS 0.097** for the matching recorded-action
samples. This is evidence that the checkpoint responds to actions rather than
only extrapolating pixels, although two controls are not a complete causal
evaluation.

- [Continuous shelf, zero actions](iter_000003000/zero_action_controls/0000_merged.mp4)
- [Arrange set 2, zero actions](iter_000003000/zero_action_controls/0001_merged.mp4)

## Long-horizon results

The long-horizon run covers three task families and uses five chained native
windows per sample (61 frames total). Its videos and metrics are written to
[`iter_000003000/long_horizon_61f`](iter_000003000/long_horizon_61f) by
[`scripts/run_jokeru_long_horizon.sh`](../scripts/run_jokeru_long_horizon.sh).
Aggregate over the three rollouts: **PSNR 23.615**, **SSIM 0.903**, **LPIPS
0.120**.

| # | Dataset/task | Chunks | Frames | PSNR | SSIM | LPIPS | Comparison |
|---:|---|---:|---:|---:|---:|---:|---|
| 00 | continuous shelf organizing | 5 | 61 | 21.056 | 0.865 | 0.144 | [video](iter_000003000/long_horizon_61f/0000_merged.mp4) |
| 01 | arrange orange juice and green tea 3 | 5 | 61 | 25.899 | 0.925 | 0.098 | [video](iter_000003000/long_horizon_61f/0001_merged.mp4) |
| 02 | pick right purple box and place in middle | 5 | 61 | 23.891 | 0.919 | 0.120 | [video](iter_000003000/long_horizon_61f/0002_merged.mp4) |

All three action arrays have 60 active (non-zero) steps. A first/middle/last
frame inspection shows that the scene and robot geometry remain coherent
across all five chunks. The continuous-shelf sample accumulates the largest
appearance error, while the arrange sample remains closest to ground truth.

## Reproduce

Run on an available GPU:

```bash
CUDA_VISIBLE_DEVICES=0 bash scripts/run_jokeru_long_horizon.sh
```

The script writes normal runtime output under the ignored `results/` tree and
copies the final compact artifacts into this versioned directory.
