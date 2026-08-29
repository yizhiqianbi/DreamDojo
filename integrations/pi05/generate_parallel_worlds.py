#!/usr/bin/env python3
"""Generate one DreamDojo future video for each sampled π0.5 action chunk."""

from __future__ import annotations

import argparse
import json
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


def main() -> None:
    repo_root = REPO_ROOT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=repo_root / "inference_results/pi05_parallel_worlds/manifest.json",
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
    parser.add_argument("--fps", type=int, default=7)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    output_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    initial = _prepare_initial_frame(output_dir / manifest["observation_image"])
    video_input = torch.cat([initial, torch.zeros_like(initial).repeat(12, 1, 1, 1)], dim=0)
    video_input = video_input.unsqueeze(0).permute(0, 2, 1, 3, 4)
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
            if output_path.exists():
                universe["video"] = output_path.name
                continue
            actions = np.load(output_dir / universe["dreamdojo_actions"])
            video = generator.generate_vid2world(
                prompt="",
                input_path=video_input,
                action=torch.from_numpy(actions).float(),
                guidance=0,
                num_video_frames=13,
                num_latent_conditional_frames=1,
                resolution="480,640",
                seed=int(universe["seed"]),
                negative_prompt=DEFAULT_NEGATIVE_PROMPT,
                lam_video=None,
            )
            normalized = (video + 1.0) / 2.0
            frames = (
                torch.clamp(normalized[0], 0, 1)
                .mul(255)
                .to(torch.uint8)
                .permute(1, 2, 3, 0)
                .cpu()
                .numpy()
            )
            mediapy.write_video(str(output_path), frames, fps=args.fps)
            universe["video"] = output_path.name
            manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"Generated universe {universe['id']} -> {output_path}")
    finally:
        generator.cleanup()
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")


if __name__ == "__main__":
    init_environment()
    try:
        main()
    finally:
        cleanup_environment()
