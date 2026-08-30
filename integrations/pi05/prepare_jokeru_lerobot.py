#!/usr/bin/env python3
"""Build one π0.5-ready LeRobot view over all local Jokeru datasets.

The 29 FPS source videos are hard-linked. The 28/30 FPS files are stream-remuxed
(no re-encoding) onto a 29 FPS timebase so frame 4*i is synchronized with the
common 7.25 Hz control clock. Rows are sampled every four source frames, matching
DreamDojo's Jokeru cadence.
The output ``action`` at row t is the absolute command for row t+1; consequently
OpenPI's configured action horizon represents future DreamDojo transitions.
"""

from __future__ import annotations

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import uuid

import numpy as np
import pandas as pd


SOURCE_FPS = 29
STRIDE = 4
OUTPUT_FPS = SOURCE_FPS / STRIDE
ACTION_DIM = 30
VIDEO_KEYS = (
    "observation.images.left_eye",
    "observation.images.right_eye",
    "observation.images.left_wrist",
    "observation.images.right_wrist",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")


def _write_jsonl(path: Path, values: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(value) + "\n" for value in values))


def _pad(values: list[np.ndarray] | np.ndarray, source_dim: int) -> np.ndarray:
    array = np.stack(values).astype(np.float32, copy=False)
    if source_dim == ACTION_DIM:
        return array
    if source_dim != 15:
        raise ValueError(f"Expected a 15D or 30D Jokeru vector, got {source_dim}D")
    output = np.zeros((array.shape[0], ACTION_DIM), dtype=np.float32)
    output[:, :source_dim] = array
    return output


def _feature_stats(array: np.ndarray) -> dict[str, list[float] | list[int]]:
    values = np.asarray(array)
    if values.ndim == 1:
        values = values[:, None]
    return {
        "min": np.min(values, axis=0).tolist(),
        "max": np.max(values, axis=0).tolist(),
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def _materialize_video(source: Path, destination: Path, source_fps: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source_fps == SOURCE_FPS:
        try:
            os.link(source, destination)
        except OSError:
            destination.symlink_to(os.path.relpath(source, destination.parent))
        return

    # Remux only: normalize packet timestamps to 29 FPS without decoding or
    # re-encoding. Frame 4*i then lands exactly at i / 7.25 seconds.
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-itsscale",
            str(source_fps / SOURCE_FPS),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-c",
            "copy",
            "-an",
            str(destination),
        ],
        check=True,
    )


def build(
    source_root: Path,
    output_root: Path,
    *,
    overwrite: bool = False,
    video_workers: int = 8,
) -> None:
    sources = sorted(path for path in source_root.iterdir() if (path / "meta/info.json").is_file())
    if not sources:
        raise FileNotFoundError(f"No LeRobot datasets found below {source_root}")
    if output_root.exists() and not overwrite:
        print(f"π0.5 dataset already exists: {output_root}")
        return

    staging = output_root.parent / f".{output_root.name}.building-{uuid.uuid4().hex[:8]}"
    staging.mkdir(parents=True)
    all_states: list[np.ndarray] = []
    all_actions: list[np.ndarray] = []
    all_current_actions: list[np.ndarray] = []
    all_timestamps: list[np.ndarray] = []
    all_frame_indices: list[np.ndarray] = []
    all_episode_indices: list[np.ndarray] = []
    all_indices: list[np.ndarray] = []
    all_task_indices: list[np.ndarray] = []
    output_episodes: list[dict] = []
    source_manifest: list[dict] = []
    tasks: list[str] = []
    task_to_index: dict[str, int] = {}
    global_episode = 0
    global_index = 0
    video_executor = ThreadPoolExecutor(max_workers=video_workers)
    video_futures = []

    try:
        template_info = _read_json(sources[0] / "meta/info.json")
        for source in sources:
            info = _read_json(source / "meta/info.json")
            source_fps = int(info["fps"])
            source_dim = int(info["features"]["action"]["shape"][0])
            source_tasks = {int(item["task_index"]): item["task"] for item in _read_jsonl(source / "meta/tasks.jsonl")}
            episodes = _read_jsonl(source / "meta/episodes.jsonl")

            for episode in episodes:
                source_episode = int(episode["episode_index"])
                source_chunk = source_episode // int(info["chunks_size"])
                parquet_path = source / info["data_path"].format(
                    episode_chunk=source_chunk,
                    episode_index=source_episode,
                )
                frame_data = pd.read_parquet(parquet_path)
                selected = np.arange(0, len(frame_data), STRIDE, dtype=np.int64)
                selected_data = frame_data.iloc[selected].copy().reset_index(drop=True)

                states = _pad(selected_data["observation.state"].tolist(), source_dim)
                current_actions = _pad(selected_data["action"].tolist(), source_dim)
                next_indices = np.minimum(selected + STRIDE, len(frame_data) - 1)
                future_actions = _pad(frame_data.iloc[next_indices]["action"].tolist(), source_dim)

                local_tasks = [source_tasks[int(index)] for index in selected_data["task_index"].tolist()]
                mapped_tasks = []
                for task in local_tasks:
                    if task not in task_to_index:
                        task_to_index[task] = len(tasks)
                        tasks.append(task)
                    mapped_tasks.append(task_to_index[task])

                length = len(selected_data)
                frame_indices = np.arange(length, dtype=np.int64)
                timestamps = (frame_indices / OUTPUT_FPS).astype(np.float32)
                indices = np.arange(global_index, global_index + length, dtype=np.int64)
                episode_indices = np.full(length, global_episode, dtype=np.int64)
                task_indices = np.asarray(mapped_tasks, dtype=np.int64)

                selected_data["action"] = list(future_actions)
                selected_data["observation.state"] = list(states)
                selected_data["observation.current_action"] = list(current_actions)
                selected_data["timestamp"] = timestamps
                selected_data["frame_index"] = frame_indices
                selected_data["episode_index"] = episode_indices
                selected_data["index"] = indices
                selected_data["task_index"] = task_indices

                output_chunk = global_episode // 1000
                output_parquet = staging / f"data/chunk-{output_chunk:03d}/episode_{global_episode:06d}.parquet"
                output_parquet.parent.mkdir(parents=True, exist_ok=True)
                selected_data.to_parquet(output_parquet, index=False)

                for video_key in VIDEO_KEYS:
                    source_video = source / info["video_path"].format(
                        episode_chunk=source_chunk,
                        video_key=video_key,
                        episode_index=source_episode,
                    )
                    output_video = staging / template_info["video_path"].format(
                        episode_chunk=output_chunk,
                        video_key=video_key,
                        episode_index=global_episode,
                    )
                    video_futures.append(
                        video_executor.submit(
                            _materialize_video,
                            source_video,
                            output_video,
                            source_fps,
                        )
                    )

                episode_tasks = list(dict.fromkeys(local_tasks))
                output_episodes.append(
                    {"episode_index": global_episode, "tasks": episode_tasks, "length": length}
                )
                source_manifest.append(
                    {
                        "episode_index": global_episode,
                        "dataset": source.name,
                        "source_episode_index": source_episode,
                        "source_action_dim": source_dim,
                        "source_fps": source_fps,
                        "stride": STRIDE,
                    }
                )
                all_states.append(states)
                all_actions.append(future_actions)
                all_current_actions.append(current_actions)
                all_timestamps.append(timestamps)
                all_frame_indices.append(frame_indices)
                all_episode_indices.append(episode_indices)
                all_indices.append(indices)
                all_task_indices.append(task_indices)
                global_episode += 1
                global_index += length
                print(f"[{global_episode:04d}] {source.name}/episode_{source_episode:06d} -> {length} rows")

        for completed, future in enumerate(as_completed(video_futures), start=1):
            future.result()
            if completed % 200 == 0 or completed == len(video_futures):
                print(f"Materialized {completed}/{len(video_futures)} videos")
        video_executor.shutdown(wait=True)

        features = dict(template_info["features"])
        features["action"] = {"dtype": "float32", "shape": [ACTION_DIM], "names": None}
        features["observation.state"] = {"dtype": "float32", "shape": [ACTION_DIM], "names": None}
        features["observation.current_action"] = {
            "dtype": "float32",
            "shape": [ACTION_DIM],
            "names": None,
        }
        info = {
            **template_info,
            "codebase_version": "v2.0",
            "robot_type": "jokeru_humanoid",
            "total_episodes": global_episode,
            "total_frames": global_index,
            "total_tasks": len(tasks),
            "fps": OUTPUT_FPS,
            "splits": {"train": f"0:{global_episode}"},
            "features": features,
            "total_chunks": math.ceil(global_episode / 1000),
            "total_videos": global_episode * len(VIDEO_KEYS),
        }
        stats_arrays = {
            "action": np.concatenate(all_actions),
            "observation.state": np.concatenate(all_states),
            "observation.current_action": np.concatenate(all_current_actions),
            "timestamp": np.concatenate(all_timestamps),
            "frame_index": np.concatenate(all_frame_indices),
            "episode_index": np.concatenate(all_episode_indices),
            "index": np.concatenate(all_indices),
            "task_index": np.concatenate(all_task_indices),
        }
        _write_json(staging / "meta/info.json", info)
        _write_jsonl(
            staging / "meta/tasks.jsonl",
            [{"task_index": index, "task": task} for index, task in enumerate(tasks)],
        )
        _write_jsonl(staging / "meta/episodes.jsonl", output_episodes)
        _write_jsonl(staging / "meta/jokeru_sources.jsonl", source_manifest)
        _write_json(staging / "meta/stats.json", {key: _feature_stats(value) for key, value in stats_arrays.items()})

        if output_root.exists():
            backup = output_root.parent / f".{output_root.name}.old-{uuid.uuid4().hex[:8]}"
            output_root.rename(backup)
            staging.rename(output_root)
            shutil.rmtree(backup)
        else:
            staging.rename(output_root)
        print(f"Built {global_episode} episodes / {global_index} frames at {output_root}")
    except Exception:
        video_executor.shutdown(wait=False, cancel_futures=True)
        shutil.rmtree(staging, ignore_errors=True)
        raise


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, default=repo_root / "datasets/jokeru")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=repo_root / "datasets/pi05_lerobot/jokeru/pi05_world_model",
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--video-workers", type=int, default=8)
    args = parser.parse_args()
    build(
        args.source_root.resolve(),
        args.output_root.resolve(),
        overwrite=args.overwrite,
        video_workers=args.video_workers,
    )


if __name__ == "__main__":
    main()
