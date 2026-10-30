"""Heavy protocol selection and explicit checkpoint identity contracts."""
from pathlib import Path
from typing import Any

import chz
import pytest
import torch

from tests.fixtures.fake_config import fake_train_cfg
from video3d.evaluate import heavy_epoch_config, heavy_epoch_selected, load_run
from video3d.models.adapter import build_adapter
from video3d.storage import atomic_json
from video3d.config import config_dict


def test_heavy_epochs_include_only_selected_and_final() -> None:
    assert heavy_epoch_selected([1, -1], 1, 5)
    assert heavy_epoch_selected([1, -1], 5, 5)
    assert not heavy_epoch_selected([1, -1], 2, 5)
    assert not heavy_epoch_selected([], 5, 5)


def test_heavy_protocol_is_separate_from_quick(fake_root: Any, tmp_path: Path) -> None:
    cfg = fake_train_cfg(fake_root, tmp_path)
    cfg = chz.replace(cfg, val=chz.replace(cfg.val, heavy_data_root='/data/gso200'))
    heavy = heavy_epoch_config(cfg, tmp_path, 1)
    assert not cfg.eval.refined_mesh
    assert heavy.eval.refined_mesh
    assert heavy.sampler.mode == 'flow' and heavy.sampler.steps == 25
    assert heavy.eval.refinement_steps == 25
    assert heavy.eval.metrics.max_points == 20000
    assert heavy.eval.metrics.icp_iterations == 100
    assert heavy.eval.surface_samples == 8192
    assert heavy.paths.data_root == '/data/gso200'
    assert heavy.eval.evaluation_id.endswith('epoch01-heavy25')
    assert heavy.eval.evaluation_id != heavy_epoch_config(cfg, tmp_path, 5).eval.evaluation_id


def test_heavy_protocol_requires_explicit_dataset(fake_root: Any, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match='heavy_data_root'):
        heavy_epoch_config(fake_train_cfg(fake_root, tmp_path), tmp_path, 1)


def test_explicit_epoch_checkpoint_wins_over_best(fake_root: Any, tmp_path: Path) -> None:
    cfg = fake_train_cfg(fake_root, tmp_path)
    run = tmp_path / 'run'
    run.mkdir()
    atomic_json(run / 'config.json', config_dict(cfg))
    adapter = build_adapter(cfg.adapter)
    torch.save({'ema':adapter.state_dict()}, run / 'epoch_01.pt')
    torch.save({'ema':{}}, run / 'best.pt')
    _, loaded, _ = load_run(cfg, str(run), torch.device('cpu'), str(run / 'epoch_01.pt'))
    assert loaded is not None
    assert loaded.state_dict().keys() == adapter.state_dict().keys()
