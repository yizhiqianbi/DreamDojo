#!/usr/bin/env python3
"""Build official student-inference inputs from the 8 π0.5 parallel branches."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest",
        type=Path,
        default=REPO_ROOT / "inference_results/pi05_parallel_worlds_48_x8/manifest.json",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=REPO_ROOT / "outputs/self_forcing/jokeru/eval/pi05_parallel_x8.json",
    )
    parser.add_argument(
        "--video-output-dir",
        type=Path,
        default=REPO_ROOT / "outputs/self_forcing/jokeru/eval/videos",
    )
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    artifact_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    observation = (artifact_dir / manifest["observation_video"]).resolve()
    entries = []
    for universe in manifest["universes"][:8]:
        entries.append(
            {
                "input_video": str(observation),
                "input_action": str((artifact_dir / universe["dreamdojo_actions"]).resolve()),
                "output_video": str((args.video_output_dir / f"universe_{int(universe['id']):02d}.mp4").resolve()),
                "resolution": [480, 640],
                "start_frame_idx": 0,
                "seed": int(universe["seed"]),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.video_output_dir.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(entries, indent=2) + "\n")
    print(f"Wrote {len(entries)} Self-Forcing student inputs to {args.output}")


if __name__ == "__main__":
    main()
