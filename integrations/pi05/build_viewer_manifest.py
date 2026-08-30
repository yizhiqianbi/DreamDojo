#!/usr/bin/env python3
"""Add compact action traces and web asset paths to a parallel-world manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    manifest_path = args.manifest.resolve()
    artifact_dir = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    for universe in manifest["universes"]:
        actions = np.load(artifact_dir / universe["pi05_actions"]).astype(np.float32)
        magnitude = np.sqrt(np.mean(actions**2, axis=1))
        low, high = np.percentile(magnitude, [5, 95])
        normalized = np.clip((magnitude - low) / max(float(high - low), 1e-6), 0.0, 1.0)
        universe["action_trace"] = np.rint(18.0 + normalized * 82.0).astype(int).tolist()
        universe["video"] = f"universe-{int(universe['id']):02d}.mp4"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote viewer manifest to {args.output}")


if __name__ == "__main__":
    main()
