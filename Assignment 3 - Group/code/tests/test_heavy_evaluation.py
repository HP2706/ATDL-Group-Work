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
    assert heavy.eval.native_reference
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


def test_baseline_evaluation_uses_separate_logging_identity(fake_root: Any, tmp_path: Path) -> None:
    cfg = fake_train_cfg(fake_root, tmp_path)
    cfg = chz.replace(cfg, eval=chz.replace(cfg.eval, evaluation_id='independent-evaluation'))
    resolved, adapter, _ = load_run(cfg, 'zero-shot', torch.device('cpu'))
    assert adapter is None and resolved.run_name == 'independent-evaluation'


def test_native_baseline_uses_selected_reference(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image
    from video3d.baselines.api import BaselineConfig, BaselineInput, reconstruct
    frames, masks = [], []
    for index in range(2):
        frame, mask = tmp_path/f'rgb{index}.png', tmp_path/f'mask{index}.png'
        Image.fromarray(np.full((4,4,3), index*100, dtype=np.uint8)).save(frame)
        Image.fromarray(np.full((4,4), 255, dtype=np.uint8)).save(mask)
        frames.append(frame)
        masks.append(mask)
    class Pipeline:
        def run(self, **kwargs: Any) -> dict[str, Any]:
            assert np.all(kwargs['image'] == 100)
            return {'coords':torch.tensor([[0,0,0,0]])}
    sample = BaselineInput(sample_id='clip/1/16', frames=frames, masks=masks, reference_index=1)
    result = reconstruct(sample, BaselineConfig(method='sam3d'), tmp_path/'out', pipeline=Pipeline())
    assert result.selected_frames == [1]


def test_baseline_export_preserves_held_out_reference(fake_root: Any, tmp_path: Path) -> None:
    from video3d.baselines.evaluate import export_sample
    from video3d.data import build_dataset
    cfg = fake_train_cfg(fake_root, tmp_path)
    dataset = build_dataset(cfg, 'val', False, 16)
    sample, info = export_sample(dataset, 0, 'synthetic', tmp_path/'inputs')
    assert info['baseline_available_frame_ids'][sample.reference_index] == info['r']
    assert set(info['baseline_available_frame_ids']) <= set(info['frame_ids'])


def test_baseline_partial_resume_has_unique_records(fake_root: Any, tmp_path: Path,
                                                   monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    import json
    module = importlib.import_module('video3d.baselines.evaluate')
    class Pipeline:
        def run(self, **kwargs: Any) -> dict[str, Any]:
            coords = torch.cartesian_prod(torch.arange(25,29),torch.arange(25,29),torch.arange(25,29))
            return {'coords':torch.cat([torch.zeros(len(coords),1,dtype=torch.long),coords],dim=1)}
    monkeypatch.setattr(module, 'load_pipeline', lambda cfg: Pipeline())
    cfg = fake_train_cfg(fake_root, tmp_path)
    path = tmp_path/'config.json'
    atomic_json(path, config_dict(cfg))
    output = tmp_path/'baseline'
    module.evaluate(str(path),'sam3d',str(output),split='val',limit=1,k=16)
    module.evaluate(str(path),'sam3d',str(output),split='val',limit=2,k=16)
    rows = [json.loads(line) for line in (output/'evaluation.jsonl').read_text().splitlines()]
    assert len(rows) == len({row['sample_id'] for row in rows}) == 2


def test_training_recovers_selected_epoch_evaluations(fake_root: Any, tmp_path: Path,
                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    import importlib
    from video3d.train import train
    module = importlib.import_module('video3d.evaluate')
    calls: list[tuple[str,int]] = []
    monkeypatch.setattr(module,'evaluate_training_epoch',lambda cfg,run,epoch: calls.append(('heavy',epoch)))
    monkeypatch.setattr(module,'evaluate_full_validation',lambda cfg,run,epoch: calls.append(('full',epoch)))
    cfg = fake_train_cfg(fake_root,tmp_path,**{'train.epochs':1,'val.heavy_epochs':'[1,-1]',
        'val.heavy_data_root':'/data/gso200','val.final_full_validation':'true'})
    train(cfg)
    assert calls == [('heavy',1),('full',1)]
    train(cfg)
    assert calls == [('heavy',1),('full',1),('heavy',1),('full',1)]
