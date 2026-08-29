#!/usr/bin/env python3
"""Validate the local 2B checkpoint and jokeru LeRobot repositories."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "configs" / "2b_480_640_jokeru.yaml"
REQUIRED_META = ("info.json", "episodes.jsonl", "stats.json", "tasks.jsonl")
VIDEO_KEYS = {
    "observation.images.left_eye",
    "observation.images.right_eye",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
}
WAN21_VAE_SIZE = 507_609_880
WAN21_VAE_SHA256 = "38071ab59bd94681c686fa51d75a1968f64e470262043be31f7a094e442fd981"
REASON1_REVISION = "3210bec0495fdc7a8d3dbb8d58da5711eab4b423"


def validate_checkpoint() -> int:
    root = PROJECT_ROOT / "checkpoints" / "DreamDojo" / "2B_pretrain" / "iter_000140000"
    missing = [name for name in ("model", "optim", "scheduler", "trainer") if not (root / name).is_dir()]
    if missing:
        raise RuntimeError(f"Checkpoint is missing directories: {missing}")
    total = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    if total < 20_000_000_000:
        raise RuntimeError(f"Checkpoint appears incomplete: only {total:,} bytes")
    print(f"checkpoint\t{total:,} bytes")
    return total


def validate_vae() -> None:
    path = PROJECT_ROOT / "checkpoints" / "Wan2.1" / "Wan2.1_VAE.pth"
    if not path.is_file() or path.stat().st_size != WAN21_VAE_SIZE:
        raise RuntimeError(f"Wan2.1 VAE is missing or incomplete: {path}")
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != WAN21_VAE_SHA256:
        raise RuntimeError(f"Wan2.1 VAE checksum mismatch: {path}")
    print(f"wan2.1_vae\t{WAN21_VAE_SIZE:,} bytes")


def validate_condition_encoder() -> None:
    snapshot = (
        PROJECT_ROOT
        / ".cache"
        / "huggingface"
        / "hub"
        / "models--nvidia--Cosmos-Reason1-7B"
        / "snapshots"
        / REASON1_REVISION
    )
    shards = sorted(snapshot.glob("model-*-of-00004.safetensors"))
    total = sum(path.stat().st_size for path in shards)
    if len(shards) != 4 or total < 16_500_000_000:
        raise RuntimeError(f"Cosmos-Reason1-7B cache is incomplete: {snapshot}")
    print(f"condition_encoder\t{total:,} bytes")


def validate_dataset(path: Path) -> tuple[int, int, int]:
    missing = [name for name in REQUIRED_META if not (path / "meta" / name).is_file()]
    if missing:
        raise RuntimeError(f"{path.name}: missing metadata files {missing}")

    info = json.loads((path / "meta" / "info.json").read_text())
    features = info["features"]
    action_shape = features["action"]["shape"]
    state_shape = features["observation.state"]["shape"]
    expected_shape = [15] if path.name == "take_wrong_item_right_arm" else [30]
    if action_shape != expected_shape or state_shape != expected_shape:
        raise RuntimeError(
            f"{path.name}: action/state shape {action_shape}/{state_shape}, expected {expected_shape}"
        )
    present_video_keys = {key for key, value in features.items() if value.get("dtype") == "video"}
    if present_video_keys != VIDEO_KEYS:
        raise RuntimeError(f"{path.name}: unexpected video keys {sorted(present_video_keys)}")

    episodes = [
        json.loads(line)
        for line in (path / "meta" / "episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    parquet_files = list((path / "data").glob("*/*.parquet"))
    video_files = list((path / "videos").glob("*/*/*.mp4"))
    expected_episodes = info["total_episodes"]
    expected_videos = info["total_videos"]
    if len(episodes) != expected_episodes or len(parquet_files) != expected_episodes:
        raise RuntimeError(
            f"{path.name}: episodes metadata/parquet mismatch "
            f"({len(episodes)}/{len(parquet_files)}/{expected_episodes})"
        )
    if len(video_files) != expected_videos:
        raise RuntimeError(f"{path.name}: videos {len(video_files)} != expected {expected_videos}")

    size = sum(file.stat().st_size for file in path.rglob("*") if file.is_file())
    print(f"{path.name}\t{expected_episodes} episodes\t{info['total_frames']} frames\t{size:,} bytes")
    return expected_episodes, info["total_frames"], size


def main() -> None:
    config = yaml.safe_load(CONFIG_PATH.read_text())
    configured_paths = config["dataloader_train"]["dataset"]["dataset_path"]
    if len(configured_paths) != 13 or len(set(configured_paths)) != 13:
        raise RuntimeError("Expected exactly 13 unique dataset paths in the training config")

    validate_checkpoint()
    validate_vae()
    validate_condition_encoder()
    totals = [validate_dataset(PROJECT_ROOT / relative_path) for relative_path in configured_paths]
    print(
        "total\t"
        f"{sum(item[0] for item in totals)} episodes\t"
        f"{sum(item[1] for item in totals)} frames\t"
        f"{sum(item[2] for item in totals):,} bytes"
    )


if __name__ == "__main__":
    main()
