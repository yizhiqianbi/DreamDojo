#!/usr/bin/env python3
"""Sample multiple π0.5 action chunks from one recorded Jokeru observation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess

import cv2
import numpy as np
import pandas as pd

from integrations.pi05.bridge import to_dreamdojo_actions
from openpi.policies import policy_config
from openpi.shared import download
from openpi.training import config as config_lib


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _read_frame(video_path: Path, frame_index: int) -> np.ndarray:
    capture = cv2.VideoCapture(str(video_path))
    capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ok, frame = capture.read()
    capture.release()
    if not ok:
        raise RuntimeError(f"Could not decode frame {frame_index} from {video_path}")
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)


def _write_observation_replay(
    video_path: Path,
    start_frame: int,
    output_path: Path,
    *,
    action_horizon: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    height, width = 744, 960
    crop_height = int(width / (640 / 480))
    crop_y = (height - crop_height) // 2
    frame_filter = (
        f"select='between(n,{start_frame},{start_frame + action_horizon * 4})*"
        f"not(mod(n-{start_frame},4))',setpts=4*N/29/TB,"
        f"crop={width}:{crop_height}:0:{crop_y},scale=640:480"
    )
    subprocess.run(
        [
            "ffmpeg",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-i",
            str(video_path),
            "-vf",
            frame_filter,
            "-frames:v",
            str(action_horizon + 1),
            "-r",
            "7.25",
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            str(output_path),
        ],
        check=True,
    )


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_jokeru_base_inference_48")
    parser.add_argument("--checkpoint", default="gs://openpi-assets/checkpoints/pi05_base")
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--step", type=int, default=100)
    parser.add_argument("--universes", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260830)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "datasets/pi05_lerobot/jokeru/pi05_world_model",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "inference_results/pi05_parallel_worlds_48",
    )
    args = parser.parse_args()

    dataset_root = args.dataset_root.resolve()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    info = json.loads((dataset_root / "meta/info.json").read_text())
    sources = {int(item["episode_index"]): item for item in _read_jsonl(dataset_root / "meta/jokeru_sources.jsonl")}
    tasks = {int(item["task_index"]): item["task"] for item in _read_jsonl(dataset_root / "meta/tasks.jsonl")}
    source = sources[args.episode]
    chunk = args.episode // int(info["chunks_size"])
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=args.episode,
    )
    data = pd.read_parquet(parquet_path)
    if not 0 <= args.step < len(data):
        raise IndexError(f"step={args.step} is outside episode length {len(data)}")
    row = data.iloc[args.step]
    prompt = tasks[int(row["task_index"])]
    video_paths = {
        key: dataset_root
        / info["video_path"].format(
            episode_chunk=chunk,
            video_key=key,
            episode_index=args.episode,
        )
        for key in (
            "observation.images.left_eye",
            "observation.images.left_wrist",
            "observation.images.right_wrist",
        )
    }
    source_frame = args.step * int(source["stride"])
    observation = {
        "observation/left_eye": _read_frame(video_paths["observation.images.left_eye"], source_frame),
        "observation/left_wrist": _read_frame(video_paths["observation.images.left_wrist"], source_frame),
        "observation/right_wrist": _read_frame(video_paths["observation.images.right_wrist"], source_frame),
        "observation/state": np.asarray(row["observation.state"], dtype=np.float32),
        "prompt": prompt,
    }

    config = config_lib.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    if data_config.norm_stats is None:
        raise FileNotFoundError(
            "Jokeru OpenPI norm stats are missing. Run integrations/pi05/compute_norm_stats.py first."
        )
    checkpoint = download.maybe_download(args.checkpoint)
    policy = policy_config.create_trained_policy(
        config,
        checkpoint,
        norm_stats=data_config.norm_stats,
        default_prompt=prompt,
    )

    initial_frame_path = output_dir / "observation.png"
    cv2.imwrite(
        str(initial_frame_path),
        cv2.cvtColor(observation["observation/left_eye"], cv2.COLOR_RGB2BGR),
    )
    replay_path = output_dir / "observation_replay.mp4"
    _write_observation_replay(
        video_paths["observation.images.left_eye"],
        source_frame,
        replay_path,
        action_horizon=int(config.model.action_horizon),
    )
    source_stats = repo_root / "datasets/jokeru" / source["dataset"] / "meta/stats.json"
    universes = []
    reference_actions = None
    for universe_index in range(args.universes):
        seed = args.seed + universe_index
        noise = np.random.default_rng(seed).standard_normal(
            (config.model.action_horizon, config.model.action_dim), dtype=np.float32
        )
        result = policy.infer(observation, noise=noise)
        actions = np.asarray(result["actions"], dtype=np.float32)
        dreamdojo_actions = to_dreamdojo_actions(
            actions,
            np.asarray(row["observation.current_action"], dtype=np.float32),
            source_dim=int(source["source_action_dim"]),
            stats_path=source_stats,
        )
        raw_path = output_dir / f"universe_{universe_index:02d}_pi05_actions.npy"
        condition_path = output_dir / f"universe_{universe_index:02d}_dreamdojo_actions.npy"
        np.save(raw_path, actions)
        np.save(condition_path, dreamdojo_actions)
        if reference_actions is None:
            reference_actions = actions
        divergence = float(np.linalg.norm(actions - reference_actions) / np.sqrt(actions.size))
        universes.append(
            {
                "id": universe_index,
                "seed": seed,
                "pi05_actions": raw_path.name,
                "dreamdojo_actions": condition_path.name,
                "action_rms_from_universe_0": divergence,
                "pi05_infer_ms": float(result["policy_timing"]["infer_ms"]),
                "video": None,
            }
        )

    manifest = {
        "schema_version": 1,
        "observation_mode": "recorded-video-replay",
        "action_mode": "pi05-flow-matching",
        "pi05_config": args.config,
        "pi05_checkpoint": args.checkpoint,
        "action_horizon": int(config.model.action_horizon),
        "dreamdojo_chunk_size": 12,
        "dataset": source["dataset"],
        "episode": int(source["source_episode_index"]),
        "merged_episode": args.episode,
        "step": args.step,
        "source_frame": source_frame,
        "prompt": prompt,
        "observation_image": initial_frame_path.name,
        "observation_video": replay_path.name,
        "universes": universes,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Sampled {len(universes)} π0.5 universes into {output_dir}")


if __name__ == "__main__":
    main()
