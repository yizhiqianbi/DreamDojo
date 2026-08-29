#!/usr/bin/env python3
"""Run an upstream OpenPI script with the Jokeru π0.5 configs registered."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import runpy
import sys


def main() -> None:
    repo_root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--openpi-root", type=Path, default=repo_root.parent / "openpi")
    parser.add_argument("script")
    args, forwarded = parser.parse_known_args()
    openpi_root = args.openpi_root.resolve()
    requested_script = Path(args.script)
    script = requested_script.resolve() if requested_script.is_file() else openpi_root / requested_script
    if not script.is_file():
        raise FileNotFoundError(script)

    os.environ.setdefault("HF_LEROBOT_HOME", str(repo_root / "datasets/pi05_lerobot"))
    sys.path.insert(0, str(repo_root))
    sys.path.insert(0, str(openpi_root / "src"))
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from openpi_config import register

    register()
    sys.argv = [str(script), *forwarded]
    runpy.run_path(str(script), run_name="__main__")


if __name__ == "__main__":
    main()
