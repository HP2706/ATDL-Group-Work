"""Validation reuses one reconstruction per object and records missing shape scores honestly."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch

from video3d.config import ExperimentConfig
from video3d.metrics import shape_metrics
from video3d.train import val_shape_metrics


class FixedDataset:
    def __init__(self, count: int) -> None:
        self.index = [{"clip_id": str(i), "asset_id": str(i), "k": 1, "r": 0,
                       "rules": ["reference"], "frame_ids": [0]} for i in range(count)]

    def __getitem__(self, index: int) -> dict[str, Any]:
        return {"info": self.index[index]}


class FixedSampler:
    def __init__(self, predictions: list[torch.Tensor]) -> None:
        self.predictions = predictions
        self.calls = 0

    def sample(self, condition: Any, seeds: list[int], config: Any) -> tuple[torch.Tensor, None]:
        self.calls += 1
        return self.predictions[self.calls - 1][None], None

    def decode_latent(self, latent: torch.Tensor) -> torch.Tensor:
        return latent


@pytest.mark.parametrize("empty", [[False, True], [True, True]])
def test_validation_records_empty_shapes_without_zero_chamfer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, empty: list[bool],
) -> None:
    truth = torch.ones(4, 4, 4, dtype=torch.bool)
    sampler = FixedSampler([torch.zeros_like(truth) if value else truth for value in empty])
    monkeypatch.setattr("video3d.train.adapted_condition", lambda *args: None)
    monkeypatch.setattr("video3d.data.ground_truth", lambda *args: np.asarray(truth))
    def float32_metrics(*args: Any, **kwargs: Any) -> dict[str, Any]:
        assert not torch.is_autocast_enabled("cpu")
        return shape_metrics(*args, **kwargs)
    monkeypatch.setattr("video3d.train.shape_metrics", float32_metrics)
    path = tmp_path / "validation.json"
    with torch.autocast("cpu", dtype=torch.bfloat16):
        result = val_shape_metrics(sampler, None, None, FixedDataset(2), ExperimentConfig(),
                                   torch.device("cpu"), path)
    assert sampler.calls == 2
    assert result["val_empty_predictions"] == sum(empty)
    assert result["val_chamfer_count"] == 2 - sum(empty)
    assert result["val_chamfer"] == (None if all(empty) else 0.0)
    assert result["val_viou"] == (0.0 if all(empty) else 0.5)
    saved = json.loads(path.read_text())
    assert saved["summary"] == result
    assert len(saved["objects"]) == 2
    assert saved["objects"][1]["chamfer"] is None


def test_validation_rejects_no_selected_objects() -> None:
    with pytest.raises(ValueError, match="at least one"):
        val_shape_metrics(FixedSampler([]), None, None, FixedDataset(0), ExperimentConfig(),
                           torch.device("cpu"))
