from pathlib import Path
import json

import numpy as np

from integrations.pi05.bridge import to_dreamdojo_actions


def test_grouped_delta_and_embedding(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"action": {"min": [0.0] * 30, "max": [10.0] * 30}}))
    current = np.zeros(30, dtype=np.float32)
    predicted = np.arange(1, 13, dtype=np.float32)[:, None].repeat(30, axis=1)
    output = to_dreamdojo_actions(predicted, current, source_dim=30, stats_path=stats_path)
    expected = np.asarray([0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.6, 0.8, 0.2, 0.4, 0.4, 0.4])
    # Values 11 and 12 clip at 1.0, so the final group's last deltas saturate.
    np.testing.assert_allclose(output[:, 169], expected, atol=1e-6)
    assert np.count_nonzero(output[:, :169]) == 0
    assert np.count_nonzero(output[:, 199:]) == 0


def test_48_step_horizon(tmp_path: Path) -> None:
    stats_path = tmp_path / "stats.json"
    stats_path.write_text(json.dumps({"action": {"min": [-100.0] * 30, "max": [100.0] * 30}}))
    current = np.zeros(30, dtype=np.float32)
    predicted = np.arange(1, 49, dtype=np.float32)[:, None].repeat(30, axis=1)
    output = to_dreamdojo_actions(predicted, current, source_dim=30, stats_path=stats_path)
    assert output.shape == (48, 384)
    # Each four-action group is relative to the preceding action.
    np.testing.assert_allclose(output[4:8, 169], [0.01, 0.02, 0.03, 0.04], atol=1e-6)
    np.testing.assert_allclose(output[44:48, 169], [0.01, 0.02, 0.03, 0.04], atol=1e-6)
