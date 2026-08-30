#!/usr/bin/env python3
"""Build a five-stage, GT-anchored parallel-future manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from integrations.pi05.bridge import to_dreamdojo_actions


def _read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _trace(actions: np.ndarray) -> list[int]:
    magnitude = np.sqrt(np.mean(np.asarray(actions, dtype=np.float32) ** 2, axis=1))
    low, high = np.percentile(magnitude, [5, 95])
    normalized = np.clip((magnitude - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    return np.rint(18.0 + normalized * 82.0).astype(int).tolist()


def _candidate(
    *,
    display_id: int,
    source: str,
    action_path: str,
    video_path: str,
    actions: np.ndarray,
    seed: int | None = None,
) -> dict:
    return {
        "display_id": display_id,
        "source": source,
        "seed": seed,
        "actions": action_path,
        "video": video_path,
        "action_trace": _trace(actions),
    }


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repo_root / "inference_results/pi05_gt_path_5stage",
    )
    parser.add_argument("--episode", type=int, default=0)
    parser.add_argument("--steps", type=int, nargs=5, default=(100, 148, 196, 244, 292))
    args = parser.parse_args()

    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    dataset_root = repo_root / "datasets/pi05_lerobot/jokeru/pi05_world_model"
    info = json.loads((dataset_root / "meta/info.json").read_text())
    sources = {
        int(item["episode_index"]): item
        for item in _read_jsonl(dataset_root / "meta/jokeru_sources.jsonl")
    }
    source = sources[args.episode]
    chunk = args.episode // int(info["chunks_size"])
    parquet_path = dataset_root / info["data_path"].format(
        episode_chunk=chunk,
        episode_index=args.episode,
    )
    data = pd.read_parquet(parquet_path)
    source_stats = repo_root / "datasets/jokeru" / source["dataset"] / "meta/stats.json"
    horizon = 48

    gt_slots = (4, 2, 6, 1, 5)
    segment_specs = []
    for index, (step, gt_slot) in enumerate(zip(args.steps, gt_slots, strict=True), start=1):
        first_segment = index == 1
        segment_specs.append(
            {
                "id": index,
                "step": int(step),
                "source_dir": (
                    repo_root / "inference_results/pi05_parallel_worlds_48_x8"
                    if first_segment
                    else output_dir / f"segment_{index:02d}"
                ),
                "source_prefix": (
                    "../pi05_parallel_worlds_48_x8" if first_segment else f"segment_{index:02d}"
                ),
                "policy_count": 7,
                "gt_slot": gt_slot,
            }
        )

    segments = []
    for spec in segment_specs:
        step = spec["step"]
        if step + horizon > len(data):
            raise IndexError(f"step {step} + horizon {horizon} exceeds episode length {len(data)}")
        source_dir = Path(spec["source_dir"])
        source_manifest = json.loads((source_dir / "manifest.json").read_text())
        gt_actions = np.stack(data.iloc[step : step + horizon]["action"].tolist()).astype(np.float32)
        gt_dreamdojo_actions = to_dreamdojo_actions(
            gt_actions,
            np.asarray(data.iloc[step]["observation.current_action"], dtype=np.float32),
            source_dim=int(source["source_action_dim"]),
            stats_path=source_stats,
        )
        gt_action_path = output_dir / f"segment_{spec['id']:02d}_gt_pi05_actions.npy"
        gt_condition_path = output_dir / f"segment_{spec['id']:02d}_gt_dreamdojo_actions.npy"
        np.save(gt_action_path, gt_actions)
        np.save(gt_condition_path, gt_dreamdojo_actions)

        policy_candidates = []
        for universe in source_manifest["universes"][: spec["policy_count"]]:
            action_file = source_dir / universe["pi05_actions"]
            actions = np.load(action_file)
            policy_candidates.append(
                {
                    "source": "pi05",
                    "seed": int(universe["seed"]),
                    "actions": f"{spec['source_prefix']}/{universe['pi05_actions']}",
                    "video": f"{spec['source_prefix']}/{universe['video']}",
                    "action_trace": _trace(actions),
                }
            )

        candidates = []
        policy_index = 0
        for display_id in range(8):
            if display_id == spec["gt_slot"]:
                candidates.append(
                    _candidate(
                        display_id=display_id,
                        source="ground_truth",
                        action_path=gt_action_path.name,
                        video_path=f"{spec['source_prefix']}/observation_replay.mp4",
                        actions=gt_actions,
                    )
                )
                continue
            policy = policy_candidates[policy_index]
            policy_index += 1
            candidates.append({"display_id": display_id, **policy})

        segments.append(
            {
                "id": spec["id"],
                "step": step,
                "end_step": step + horizon,
                "source_frame": int(source_manifest["source_frame"]),
                "prompt": source_manifest["prompt"],
                "observation_image": f"{spec['source_prefix']}/observation.png",
                "gt_slot": int(spec["gt_slot"]),
                "candidates": candidates,
            }
        )

    result = {
        "schema_version": 1,
        "episode": int(source["source_episode_index"]),
        "dataset": source["dataset"],
        "action_horizon": horizon,
        "policy_candidates_per_segment": 7,
        "ground_truth_candidates_per_segment": 1,
        "selection_rule": "follow-recorded-ground-truth",
        "segments": segments,
    }
    (output_dir / "manifest.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"Wrote five-stage GT path artifact to {output_dir}")


if __name__ == "__main__":
    main()
