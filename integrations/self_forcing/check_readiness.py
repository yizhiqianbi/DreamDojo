#!/usr/bin/env python3
"""Audit local Jokeru Self-Forcing artifacts without allocating a GPU."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]


def _complete_dcp(iteration_dir: Path) -> bool:
    return (iteration_dir / "model" / ".metadata").is_file()


def _cache_ids(cache_root: Path) -> set[str]:
    required = []
    for directory, suffix in (("actions", ".json"), ("images", ".png"), ("videos", ".mp4"), ("latents", ".pt")):
        required.append({path.stem for path in (cache_root / directory).glob(f"*{suffix}")})
    return set.intersection(*required) if required else set()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--teacher",
        type=Path,
        default=REPO_ROOT / "outputs/dreamdojo/jokeru_posttrain/jokeru_2b_8gpu_posttrain_20260829/checkpoints/iter_000003000",
    )
    parser.add_argument("--cache", type=Path, default=REPO_ROOT / "outputs/self_forcing/jokeru/teacher_cache")
    parser.add_argument("--warmup", type=Path, default=REPO_ROOT / "outputs/self_forcing/jokeru/warmup/checkpoints/iter_000020000")
    parser.add_argument("--student", type=Path, default=REPO_ROOT / "outputs/self_forcing/jokeru/student/checkpoints/iter_000003000")
    args = parser.parse_args()

    cache_ids = _cache_ids(args.cache)
    report = {
        "teacher_checkpoint": {"path": str(args.teacher), "ready": _complete_dcp(args.teacher)},
        "teacher_cache": {"path": str(args.cache), "complete_samples": len(cache_ids)},
        "causal_warmup_checkpoint": {"path": str(args.warmup), "ready": _complete_dcp(args.warmup)},
        "self_forcing_student_checkpoint": {"path": str(args.student), "ready": _complete_dcp(args.student)},
        "next_stage": (
            "student_inference"
            if _complete_dcp(args.student)
            else "self_forcing"
            if _complete_dcp(args.warmup)
            else "causal_warmup"
            if cache_ids
            else "teacher_cache"
        ),
    }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
