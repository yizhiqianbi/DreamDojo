"""Jokeru π0.5 transforms and configs registered at runtime into OpenPI."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import einops
import numpy as np
from typing_extensions import override

from openpi import transforms
from openpi.models import model as model_lib
from openpi.models import pi0_config
from openpi.training import config as config_lib
from openpi.training import optimizer
from openpi.training import weight_loaders


REPO_ROOT = Path(__file__).resolve().parents[2]
DATASET_REPO_ID = "jokeru/pi05_world_model"
ASSETS_ROOT = REPO_ROOT / "outputs/pi05/assets"
CHECKPOINT_ROOT = REPO_ROOT / "outputs/pi05/checkpoints"


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255.0, 0, 255).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image


@dataclasses.dataclass(frozen=True)
class JokeruInputs(transforms.DataTransformFn):
    model_type: model_lib.ModelType

    def __call__(self, data: dict) -> dict:
        inputs = {
            "state": np.asarray(data["observation/state"], dtype=np.float32),
            "image": {
                "base_0_rgb": _parse_image(data["observation/left_eye"]),
                "left_wrist_0_rgb": _parse_image(data["observation/left_wrist"]),
                "right_wrist_0_rgb": _parse_image(data["observation/right_wrist"]),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                "right_wrist_0_rgb": np.True_,
            },
        }
        if "actions" in data:
            inputs["actions"] = np.asarray(data["actions"], dtype=np.float32)
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]
        return inputs


@dataclasses.dataclass(frozen=True)
class JokeruOutputs(transforms.DataTransformFn):
    def __call__(self, data: dict) -> dict:
        return {"actions": np.asarray(data["actions"][..., :30])}


@dataclasses.dataclass(frozen=True)
class LeRobotJokeruDataConfig(config_lib.DataConfigFactory):
    @override
    def create(self, assets_dirs: Path, model_config: model_lib.BaseModelConfig) -> config_lib.DataConfig:
        repack = transforms.Group(
            inputs=[
                transforms.RepackTransform(
                    {
                        "observation/left_eye": "observation.images.left_eye",
                        "observation/left_wrist": "observation.images.left_wrist",
                        "observation/right_wrist": "observation.images.right_wrist",
                        "observation/state": "observation.state",
                        "actions": "action",
                        "prompt": "prompt",
                    }
                )
            ]
        )
        data_transforms = transforms.Group(
            inputs=[JokeruInputs(model_type=model_config.model_type)],
            outputs=[JokeruOutputs()],
        ).push(
            inputs=[transforms.DeltaActions(transforms.make_bool_mask(30))],
            outputs=[transforms.AbsoluteActions(transforms.make_bool_mask(30))],
        )
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack,
            data_transforms=data_transforms,
            model_transforms=config_lib.ModelTransformFactory()(model_config),
            action_sequence_keys=("action",),
        )


def _data_config() -> LeRobotJokeruDataConfig:
    return LeRobotJokeruDataConfig(
        repo_id=DATASET_REPO_ID,
        base_config=config_lib.DataConfig(prompt_from_task=True),
        assets=config_lib.AssetsConfig(
            assets_dir=str(ASSETS_ROOT / "pi05_jokeru_lora"),
            asset_id=DATASET_REPO_ID,
        ),
    )


def get_configs() -> list[config_lib.TrainConfig]:
    lora_model = pi0_config.Pi0Config(
        pi05=True,
        action_dim=32,
        action_horizon=12,
        paligemma_variant="gemma_2b_lora",
        action_expert_variant="gemma_300m_lora",
    )
    common = {
        "assets_base_dir": str(ASSETS_ROOT),
        "checkpoint_base_dir": str(CHECKPOINT_ROOT),
        "num_workers": 8,
        "save_interval": 1000,
        "keep_period": 5000,
        "wandb_enabled": False,
    }
    return [
        config_lib.TrainConfig(
            name="pi05_jokeru_lora",
            model=lora_model,
            data=_data_config(),
            weight_loader=weight_loaders.CheckpointWeightLoader(
                "gs://openpi-assets/checkpoints/pi05_base/params"
            ),
            freeze_filter=lora_model.get_freeze_filter(),
            ema_decay=None,
            batch_size=8,
            # ceil(3 * 95,359 frames / batch_size 8) = 35,760 optimizer steps.
            num_train_steps=35_760,
            lr_schedule=optimizer.CosineDecaySchedule(
                warmup_steps=1_000,
                peak_lr=5e-5,
                decay_steps=35_760,
                decay_lr=5e-6,
            ),
            optimizer=optimizer.AdamW(clip_gradient_norm=1.0),
            **common,
        ),
        config_lib.TrainConfig(
            name="pi05_jokeru_base_inference",
            model=pi0_config.Pi0Config(pi05=True, action_dim=32, action_horizon=12),
            data=_data_config(),
            batch_size=1,
            num_train_steps=1,
            **common,
        ),
    ]


def register() -> None:
    for config in get_configs():
        config_lib._CONFIGS_DICT[config.name] = config  # Runtime extension point for upstream OpenPI CLI.
