"""Full synthetic exports reuse the maintained split and epoch accounting."""
from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from experiments.conditioning_ablation.configure import configure
from tests.fixtures.make_fake_dataset import make_fake_dataset
from tests.test_scanned import fixture_dataset
from video3d.config import config_dict, load_config, read_config, replace_config
from video3d.data.samples import ClipDataset
from video3d.data.scanned import records
from video3d.data.scan_prepare import write_manifest


def test_full_synthetic_matrix_uses_asset_split_and_actual_samples(tmp_path: Path) -> None:
    data = make_fake_dataset(tmp_path/'export', size=32, splits=('train',))
    gso = tmp_path/'gso'
    fixture_dataset(gso, 'gso')
    row = next(row for row in records(gso) if row.split == 'eval')
    row.frames *= 2
    row.masks *= 2
    write_manifest(gso, [row.model_copy(update={'object_id':f'object-{i}'}) for i in range(250)])
    (gso/'audit.json').write_text(json.dumps({'objects':250}))
    base = tmp_path/'base.json'
    # Existing split hash puts sphere in validation and box in training.
    cfg = replace_config(load_config('c3'), **{'data.val_asset_frac':.5})
    base.write_text(json.dumps(config_dict(cfg)))
    root = tmp_path/'resolved'
    manifest = Path(configure(str(base), str(root), gso_root=str(gso),
                             synthetic_root=str(data/'train'), expected_clips=4, seed=42,
                             synthetic_val_asset_frac=.5, epochs=5, evaluation_objects=200))
    matrix = json.loads(manifest.read_text())
    split = json.loads((root/'data/synthetic/split.json').read_text())
    assert split['train_assets'] == ['box'] and split['validation_assets'] == ['sphere']
    assert not set(split['train_targets']) & set(split['validation_targets'])
    assert split['train_clips'] == split['validation_clips'] == 2
    assert matrix['train_samples'] == 2
    assert matrix['steps_per_epoch'] == math.ceil(matrix['train_samples']/32)
    assert matrix['train_objects'] == matrix['validation_objects'] == 1
    assert matrix['epochs'] == 5 and matrix['evaluation_objects'] == 200
    assert len(records(gso)) == 250
    for entry in matrix['entries']:
        resolved = read_config(entry['config'])
        assert resolved.data.source == 'synthetic' and resolved.seed == 42
        assert resolved.train.epochs == 5
        assert resolved.data.clip_lengths == [16] and resolved.data.clip_length_probs == [1.]
        assert len(ClipDataset(resolved, 'train', True)) == matrix['train_samples']
        assert len(ClipDataset(resolved, 'val', False)) == matrix['validation_samples']
        evaluation = read_config(entry['evaluation'])
        assert evaluation.data.source == 'gso' and evaluation.data.eval_split == 'eval'
        assert [r.object_id for r in records(Path(evaluation.paths.data_root))] == [r.object_id for r in records(gso)[:200]]
    assert (root/'data/synthetic/train').resolve() == (data/'train').resolve()
    with pytest.raises(ValueError, match='immutable'):
        configure(str(base), str(root), gso_root=str(gso), synthetic_root=str(data/'train'))
    with pytest.raises(ValueError, match='Full synthetic export'):
        configure(str(base), str(tmp_path/'too-small'), gso_root=str(gso),
                  synthetic_root=str(data/'train'), expected_clips=10000)
