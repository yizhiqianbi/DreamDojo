#!/usr/bin/env python3
"""Generate one DreamDojo future video for each sampled π0.5 action chunk."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import os
from pathlib import Path

import cv2
import mediapy
import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("HF_HOME", str(REPO_ROOT / ".cache/huggingface"))

from cosmos_oss.init import cleanup_environment, init_environment
from cosmos_predict2._src.predict2.inference.video2world import Video2WorldInference
from cosmos_predict2.config import DEFAULT_NEGATIVE_PROMPT


def _prepare_initial_frame(path: Path) -> torch.Tensor:
    frame = cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB)
    height, width = frame.shape[:2]
    target_ratio = 640 / 480
    if width / height > target_ratio:
        target_width = int(height * target_ratio)
        x0 = (width - target_width) // 2
        frame = frame[:, x0 : x0 + target_width]
    elif width / height < target_ratio:
        target_height = int(width / target_ratio)
        y0 = (height - target_height) // 2
        frame = frame[y0 : y0 + target_height]
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    return torch.from_numpy(frame.copy()).permute(2, 0, 1).unsqueeze(0).to(torch.uint8)


def _make_video_input(initial_frame: torch.Tensor, action_count: int) -> torch.Tensor:
    video = torch.cat(
        [initial_frame, torch.zeros_like(initial_frame).repeat(action_count, 1, 1, 1)],
        dim=0,
    )
    return video.unsqueeze(0).permute(0, 2, 1, 3, 4)


def _to_uint8_frames(video: torch.Tensor) -> np.ndarray:
    return (
        torch.clamp((video + 1.0) / 2.0, 0, 1)[0]
        .mul(255)
        .to(torch.uint8)
        .permute(1, 2, 3, 0)
        .cpu()
        .numpy()
    )


def _read_video(path: Path) -> np.ndarray:
    capture = cv2.VideoCapture(str(path))
    frames = []
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    capture.release()
    if not frames:
        raise RuntimeError(f"Could not decode generated video: {path}")
    return np.stack(frames).astype(np.float32) / 255.0


def _pairwise_differences(manifest: dict, output_dir: Path) -> dict:
    differences = {}
    for left, right in itertools.combinations(manifest["universes"], 2):
        left_video = _read_video(output_dir / left["video"])
        right_video = _read_video(output_dir / right["video"])
        if left_video.shape != right_video.shape:
            raise ValueError(f"Video shape mismatch: {left_video.shape} != {right_video.shape}")
        mse = float(np.mean((left_video - right_video) ** 2))
        differences[f"u{left['id']}_u{right['id']}"] = {
            "pixel_psnr_db": float(-10.0 * math.log10(max(mse, 1e-12))),
            "last_frame_mae": float(np.mean(np.abs(left_video[-1] - right_video[-1]))),
        }
    return differences


def main() -> None:
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "inference_results/pi05_parallel_worlds_48/manifest.json",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=(
            repo_root
            / "outputs/dreamdojo/jokeru_posttrain/jokeru_2b_8gpu_posttrain_20260829/checkpoints/iter_000003000/model_ema_bf16.pt"
        ),
    )
    parser.add_argument("--experiment", default="dreamdojo_2b_480_640_jokeru")
    parser.add_argument("--config-file", default="cosmos_predict2/_src/predict2/action/configs/action_conditioned/config.py")
    parser.add_argument("--fps", type=float, default=7.25)
    parser.add_argument("--chunk-size", type=int, default=12)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    initial = _prepare_initial_frame(output_dir / manifest["observation_image"])
    generator = Video2WorldInference(
        experiment_name=args.experiment,
        ckpt_path=args.checkpoint.resolve(),
        s3_credential_path="",
        context_parallel_size=1,
        config_file=args.config_file,
    )
    try:
        universes = manifest["universes"] if args.limit is None else manifest["universes"][: args.limit]
        for universe in universes:
            output_path = output_dir / f"universe_{int(universe['id']):02d}.mp4"
            if output_path.exists() and not args.overwrite:
                universe["video"] = output_path.name
                continue
            actions = np.load(output_dir / universe["dreamdojo_actions"])
            if len(actions) % args.chunk_size:
                raise ValueError(
                    f"Action horizon {len(actions)} must be divisible by chunk size {args.chunk_size}"
                )
            current_frame = initial
            rollout_frames = []
            chunk_records = []
            for chunk_index, start in enumerate(range(0, len(actions), args.chunk_size)):
                chunk_actions = actions[start : start + args.chunk_size]
                chunk_seed = int(universe["seed"]) + chunk_index * 10_000
                video = generator.generate_vid2world(
                    prompt="",
                    input_path=_make_video_input(current_frame, len(chunk_actions)),
                    action=torch.from_numpy(chunk_actions).float(),
                    guidance=0,
                    num_video_frames=len(chunk_actions) + 1,
                    num_latent_conditional_frames=1,
                    resolution="480,640",
                    seed=chunk_seed,
                    negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                    lam_video=None,
                )
                frames = _to_uint8_frames(video)
                rollout_frames.extend(frames if chunk_index == 0 else frames[1:])
                current_frame = torch.from_numpy(frames[-1].copy()).permute(2, 0, 1).unsqueeze(0)
                chunk_records.append(
                    {
                        "index": chunk_index,
                        "action_start": start,
                        "action_end": start + len(chunk_actions),
                        "seed": chunk_seed,
                    }
                )
                print(
                    f"Universe {universe['id']} chunk {chunk_index + 1}/{len(actions) // args.chunk_size} "
                    f"actions [{start}:{start + len(chunk_actions)}]"
                )
            mediapy.write_video(str(output_path), np.stack(rollout_frames), fps=args.fps)
            universe["video"] = output_path.name
            universe["dreamdojo_chunks"] = chunk_records
            universe["generated_video_frames"] = len(rollout_frames)
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Generated universe {universe['id']} -> {output_path}")
    finally:
        generator.cleanup()
    if args.limit is None:
        manifest["pairwise_video_difference"] = _pairwise_differences(manifest, output_dir)
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    init_environment()
    try:
        main()
    finally:
        cleanup_environment()
