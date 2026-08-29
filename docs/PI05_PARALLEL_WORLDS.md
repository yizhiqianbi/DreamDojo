# π0.5 fine-tuning and DreamDojo parallel worlds

This integration connects the upstream Physical Intelligence OpenPI
implementation of π0.5 to the Jokeru post-trained DreamDojo 2B checkpoint.
Given the same recorded observation and prompt, π0.5 samples several action
chunks. Each chunk conditions an independent DreamDojo future, producing the
parallel-universe visualization in the
[private viewer](https://dreamdojo-jokeru-lab.boingshaw.chatgpt.site).

## What is implemented

```text
recorded Jokeru observation + task text
                  |
                  v
          π0.5 flow policy
        /          |          \
  seed A        seed B       seed C       (12 × 30 absolute actions)
     |             |            |
     +--- exact Jokeru -> DreamDojo bridge ---+   (12 × 384 deltas)
     |             |            |
 DreamDojo A   DreamDojo B  DreamDojo C       (three 13-frame futures)
```

The mock control semantics are intentional. At control step `t`, all policy
branches receive the same cameras/state copied from the prerecorded episode.
The generated videos visualize counterfactual futures; they are not fed back
as the observation at `t+1`. Replacing the replay source with live robot input
later does not change the policy or world-model interfaces.

Multiple actions are obtained through OpenPI's supported `Policy.infer(...,
noise=...)` interface. Every branch has an independent seeded Gaussian initial
noise tensor for the flow-matching sampler. No manual perturbation is added to
the resulting action.

## Jokeru data adapter

`prepare_jokeru_lerobot.py` merges every local Jokeru dataset into one LeRobot
view at `datasets/pi05_lerobot/jokeru/pi05_world_model`:

- 13 datasets, 592 episodes, 95,359 training rows, and 6 task prompts.
- 548 episodes with 30D actions and 44 right-arm episodes with 15D actions;
  15D vectors are zero-padded to 30D while source dimensionality is retained in
  the provenance manifest.
- Four views remain in the dataset. π0.5 consumes left eye, left wrist, and
  right wrist because its standard observation contract has three image slots.
- Rows use every fourth source frame. The common control rate is 7.25 Hz.
  Source videos at 28 or 30 FPS are timestamp-remuxed to a 29 FPS timebase
  without re-encoding; 29 FPS files are hard-linked.
- The action stored at row `t` is the absolute command at source frame `t+4`,
  so the 12-action policy horizon corresponds to the next 12 DreamDojo
  transitions.

Build the view and quantile normalization assets:

```bash
python integrations/pi05/prepare_jokeru_lerobot.py
python integrations/pi05/compute_norm_stats.py
```

The generated dataset and normalization assets are ignored by Git because they
are derived local artifacts.

## π0.5 LoRA configuration

`pi05_jokeru_lora` applies LoRA to the PaliGemma 2B backbone and 300M action
expert, uses 12-step chunks padded to OpenPI's 32D model action width, and trains
with batch size 8. It is configured for 35,760 optimizer steps:

```text
ceil(3 epochs × 95,359 rows / batch 8) = 35,760 steps
```

Checkpoints are saved every 1,000 steps and retained every 5,000 steps. Run
from this DreamDojo repository while the official OpenPI clone is its sibling:

```bash
git clone https://github.com/Physical-Intelligence/openpi ../openpi
GIT_LFS_SKIP_SMUDGE=1 uv sync --project ../openpi

CUDA_VISIBLE_DEVICES=0 \
XLA_PYTHON_CLIENT_MEM_FRACTION=0.9 \
uv run --project ../openpi python integrations/pi05/run_openpi.py \
  --openpi-root ../openpi \
  scripts/train.py pi05_jokeru_lora \
  --exp-name=jokeru_3ep \
  --no-wandb-enabled
```

For multi-GPU JAX training, expose the desired devices and keep the global
batch divisible by the JAX device count. The default batch 8 is suitable for
eight devices when capacity is available.

The adapter was validated against a real transformed OpenPI sample: state
`[32]`, actions `[12, 32]`, three images `[224, 224, 3]`, and 200 prompt tokens.
A real batch-1, one-step LoRA smoke run restored the 12.5 GiB base parameters,
completed with loss `0.8082`, gradient norm `3.6176`, parameter norm
`1803.8630`, and wrote a restorable checkpoint. This proves the path runs; it
is not a trained policy checkpoint.

## Action bridge

π0.5 emits 12 absolute Jokeru controls. `bridge.py` converts them to the exact
conditioning used during DreamDojo Jokeru post-training:

1. Clip/min-max normalize with the source dataset's action statistics.
2. Form four-step grouped deltas. Group 0 uses the recorded current action as
   baseline; groups 1 and 2 use the preceding predicted action.
3. Embed 30D controls into DreamDojo indices `[169:199]`, or 15D right-arm
   controls into `[199:214]`, in a `[12, 384]` tensor.

## Sample and generate parallel worlds

The full launcher prepares data/stats if necessary, samples π0.5 actions, and
then loads DreamDojo once to generate every branch:

```bash
CUDA_VISIBLE_DEVICES=0 \
bash scripts/run_pi05_parallel_worlds.sh \
  --episode 0 --step 100 --universes 3 --seed 20260830
```

To use a completed fine-tuned policy, point `PI05_CHECKPOINT` at its checkpoint
directory and pass `--config pi05_jokeru_lora`. The committed example instead
uses the official `pi05_base` checkpoint so its provenance is unambiguous.

The committed run produced warm π0.5 samples in 61.9–78.4 ms after a 14.1 s
JIT cold start. Action RMS divergence from branch A is 0.0414 and 0.0553.
Pairwise generated-video PSNR is 18.30–20.02 dB, showing materially different
DreamDojo futures from the fixed observation. See
[`inference_results/pi05_parallel_worlds`](../inference_results/pi05_parallel_worlds).

## Key files

- `integrations/pi05/openpi_config.py`: runtime OpenPI train/inference configs.
- `integrations/pi05/prepare_jokeru_lerobot.py`: all-dataset LeRobot adapter.
- `integrations/pi05/sample_pi05_actions.py`: replay observation and multi-seed
  policy sampling.
- `integrations/pi05/bridge.py`: absolute VLA actions to DreamDojo condition.
- `integrations/pi05/generate_parallel_worlds.py`: one world-model future per
  action branch.
- `viewer/app/page.tsx`: interactive source/action/future visualization.
