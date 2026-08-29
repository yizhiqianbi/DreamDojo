#!/usr/bin/env python3
"""Load and transform one Jokeru example through OpenPI's official data path."""

from __future__ import annotations

import argparse
import json

import numpy as np

from openpi.training import config as config_lib
from openpi.training import data_loader


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="pi05_jokeru_lora")
    parser.add_argument("--index", type=int, default=100)
    args = parser.parse_args()
    config = config_lib.get_config(args.config)
    data_config = config.data.create(config.assets_dirs, config.model)
    dataset = data_loader.create_torch_dataset(data_config, config.model.action_horizon, config.model)
    dataset = data_loader.transform_dataset(dataset, data_config)
    item = dataset[args.index]
    summary = {
        "dataset_length": len(dataset),
        "state": list(np.asarray(item["state"]).shape),
        "actions": list(np.asarray(item["actions"]).shape),
        "images": {key: list(np.asarray(value).shape) for key, value in item["image"].items()},
        "tokenized_prompt": list(np.asarray(item["tokenized_prompt"]).shape),
        "action_finite": bool(np.isfinite(np.asarray(item["actions"])).all()),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
