# jokeru post-training

This adaptation trains the DreamDojo 2B model on all public datasets owned by
`jokeru` on ModelScope.

## Data mapping

The 13 repositories use LeRobot v2.1 and four camera streams. Twelve use a
30-dimensional `action` and `observation.state`; `take_wrong_item_right_arm`
uses a distinct 15-dimensional control schema. Training uses
`observation.images.left_eye` as the primary egocentric view and samples the
28--30 FPS source at a stride of four.

DreamDojo's existing action layout is preserved. The 30-dimensional jokeru
schema is placed in `[169, 199)`, and the separate 15-dimensional right-arm
schema in `[199, 214)`. Both fit inside the officially reserved `[169, 220)`
range. The existing GR-1, G1, YAM, AgiBot, MANO, and latent-action ranges are
not changed.

## Download and validate

The downloader is resumable and discovers the current public dataset list from
the ModelScope API. It also downloads the public Wan2.1 VAE and the frozen
Cosmos-Reason1-7B condition encoder required by the training graph, so training
does not depend on gated Cosmos weights:

```bash
uvx --from modelscope-hub --with huggingface-hub \
  python scripts/download_jokeru_assets.py --max-workers 8

python scripts/validate_jokeru_assets.py
```

## Train

The default experiment stops at iteration 8,953, which is approximately three
effective epochs after accounting for the earlier small-batch stages, and
disables Weights & Biases logging:

```bash
NPROC=8 bash launch.sh dreamdojo_2b_480_640_jokeru
```

To reserve only physical GPU 1, use a per-device batch of one:

```bash
CUDA_VISIBLE_DEVICES=1 NPROC=1 bash launch.sh \
  dreamdojo_2b_480_640_jokeru dataloader_train.batch_size=1
```

For a long-running GPU 1 job in this shared environment, use the supervisor:

```bash
nohup setsid bash scripts/run_jokeru_gpu1.sh \
  > outputs/launch_logs/jokeru_2b_gpu1_supervisor.log 2>&1 < /dev/null &
```

The supervisor only terminates a process on physical GPU 1 when its full
command matches the known `filter_x_wv12b` inference workload. It restarts
DreamDojo after external termination, creates an early checkpoint at iteration
10, and then resumes with a 1,000-iteration checkpoint interval. DreamDojo's
same-job checkpoint discovery restores the model, optimizer, scheduler, and
iteration counter automatically.

After the single-GPU job has produced a checkpoint, migrate it to eight GPUs
with a per-GPU batch of 16 (global batch 128) using:

```bash
nohup setsid bash scripts/run_jokeru_8gpu.sh \
  > outputs/launch_logs/jokeru_2b_8gpu_supervisor.log 2>&1 < /dev/null &
```

The eight-GPU supervisor resumes the full training state from the latest
single-GPU DCP checkpoint. The launcher saves every 1,000 iterations and
retains only the latest two completed eight-GPU checkpoints.
It also removes only the known `filter_x_wv12b` interferer if that workload is
automatically relaunched on one of GPUs 0--7.

The jokeru action mapping leaves the latent-action slice at zero, so this
experiment disables the on-the-fly LAM pass as recommended for post-training.

Set `IMAGINAIRE_OUTPUT_ROOT` to override the default `outputs` directory.
Additional Hydra overrides can be appended to the command, for example
`trainer.max_iter=1` for a one-iteration smoke test.
