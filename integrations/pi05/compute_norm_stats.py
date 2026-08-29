#!/usr/bin/env python3
"""Compute OpenPI quantile stats without decoding the Jokeru videos."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _stats(values: np.ndarray) -> dict[str, list[float]]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": np.mean(values, axis=0).tolist(),
        "std": np.std(values, axis=0).tolist(),
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def compute(dataset_root: Path, output_path: Path, action_horizon: int = 12) -> None:
    info = json.loads((dataset_root / "meta/info.json").read_text())
    episodes = [
        json.loads(line)
        for line in (dataset_root / "meta/episodes.jsonl").read_text().splitlines()
        if line.strip()
    ]
    states: list[np.ndarray] = []
    action_deltas: list[np.ndarray] = []
    for episode in episodes:
        episode_index = int(episode["episode_index"])
        chunk = episode_index // int(info["chunks_size"])
        path = dataset_root / info["data_path"].format(
            episode_chunk=chunk,
            episode_index=episode_index,
        )
        data = pd.read_parquet(path, columns=["action", "observation.state"])
        state = np.stack(data["observation.state"]).astype(np.float32)
        action = np.stack(data["action"]).astype(np.float32)
        indices = np.minimum(
            np.arange(len(data))[:, None] + np.arange(action_horizon)[None, :],
            len(data) - 1,
        )
        states.append(state)
        action_deltas.append((action[indices] - state[:, None, :]).reshape(-1, action.shape[-1]))

    norm_stats = {
        "state": _stats(np.concatenate(states)),
        "actions": _stats(np.concatenate(action_deltas)),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps({"norm_stats": norm_stats}, indent=2) + "\n")
    print(f"Wrote OpenPI normalization stats to {output_path}")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repo_root / "datasets/pi05_lerobot/jokeru/pi05_world_model",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=(
            repo_root
            / "outputs/pi05/assets/pi05_jokeru_lora/jokeru/pi05_world_model/norm_stats.json"
        ),
    )
    parser.add_argument("--action-horizon", type=int, default=12)
    args = parser.parse_args()
    compute(args.dataset_root.resolve(), args.output_path.resolve(), args.action_horizon)


if __name__ == "__main__":
    main()
