"""Convert absolute π0.5 Jokeru actions to DreamDojo's 384D conditioning."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


ACTION_HORIZON = 12
DREAMDOJO_ACTION_DIM = 384


def normalize_minmax(values: np.ndarray, stats_path: Path, source_dim: int) -> np.ndarray:
    stats = json.loads(stats_path.read_text())["action"]
    minimum = np.asarray(stats["min"], dtype=np.float32)[:source_dim]
    maximum = np.asarray(stats["max"], dtype=np.float32)[:source_dim]
    values = np.asarray(values, dtype=np.float32)[..., :source_dim]
    scale = maximum - minimum
    normalized = np.zeros_like(values)
    mask = scale != 0
    normalized[..., mask] = 2.0 * (values[..., mask] - minimum[mask]) / scale[mask] - 1.0
    return np.clip(normalized, -1.0, 1.0)


def to_dreamdojo_actions(
    predicted_actions: np.ndarray,
    current_action: np.ndarray,
    *,
    source_dim: int,
    stats_path: Path,
) -> np.ndarray:
    """Match ``WrappedLeRobotSingleDataset``'s grouped-delta construction exactly."""
    predicted_actions = np.asarray(predicted_actions, dtype=np.float32)
    if predicted_actions.shape[0] != ACTION_HORIZON:
        raise ValueError(f"Expected {ACTION_HORIZON} predicted actions, got {predicted_actions.shape}")
    if source_dim not in (15, 30):
        raise ValueError(f"Unsupported Jokeru source action dimension: {source_dim}")

    predicted = normalize_minmax(predicted_actions, stats_path, source_dim)
    baseline = normalize_minmax(np.asarray(current_action)[None], stats_path, source_dim)[0]
    deltas = np.zeros_like(predicted)
    for group_start in range(0, ACTION_HORIZON, 4):
        group_baseline = baseline if group_start == 0 else predicted[group_start - 1]
        deltas[group_start : group_start + 4] = predicted[group_start : group_start + 4] - group_baseline

    output = np.zeros((ACTION_HORIZON, DREAMDOJO_ACTION_DIM), dtype=np.float32)
    if source_dim == 30:
        output[:, 169:199] = deltas
    else:
        output[:, 199:214] = deltas
    return output
