"""The experiment matrix exercises the maintained training and evaluation contracts."""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import pytest
import torch

from experiments.conditioning_ablation.configure import Variant, audited_snapshot, configure, variant_config, variants
from tests.fixtures.fake_config import FAKE_OVERRIDES
from tests.test_scanned import fixture_dataset
from video3d.config import config_dict, load_config, replace_config
from video3d.data.scanned import audit, records
from video3d.data.scan_prepare import write_manifest
from video3d.data.targets import cache_latents
from video3d.evaluate import evaluate, load_run
from video3d.models.sam3d import build_sam3d
from video3d.train import train
from video3d.utils import read_jsonl


def test_matrix_changes_only_model_switches() -> None:
    matrix = variants()
    assert len(matrix) == len({row.name for row in matrix}) == 9
    learned = [row for row in matrix if not row.baseline]
    assert [row.name for row in matrix] == ['fusion-frozen', 'fusion-sam-lora', 'sam3d-baseline',
                                              'pooler-frozen', 'pooler-sam-lora', 'dinov2-pooler', 'sam3d-random-frame-lora', 'append-frozen', 'append-sam-lora']
    assert all(not row.video_lora for row in matrix)
    assert sum(row.encoder == 'dinov2_frames' for row in learned) == 1
    base = load_config('c3')
    for row in matrix:
        cfg = variant_config(base, row, row.name)
        assert cfg.encoder.kv_source == row.encoder
        if row.strategy != "native":
            assert cfg.data == base.data
        assert cfg.sampler == base.sampler
        assert cfg.sam3d.lora == row.sam_lora
        assert cfg.encoder.vjepa.video_lora == row.video_lora
        assert cfg.adapter.strategy == row.strategy


def test_snapshot_requires_full_audited_dataset(tmp_path: Path) -> None:
    source = tmp_path/'source'
    fixture_dataset(source, 'omniobject3d')
    audit(str(source))
    with pytest.raises(ValueError, match='Full audited dataset required'):
        audited_snapshot(source, tmp_path/'snapshot', 6000)
    audited_snapshot(source, tmp_path/'snapshot', 3)
    assert (source/'manifest.jsonl').read_bytes() == (tmp_path/'snapshot/manifest.jsonl').read_bytes()


@pytest.mark.parametrize('variant', [row for row in variants() if not row.baseline], ids=lambda row: row.name)
def test_all_trainable_variants_train_resume_and_evaluate(tmp_path: Path, variant: Variant) -> None:
    omni, gso = tmp_path/'omni', tmp_path/'gso'
    fixture_dataset(omni, 'omniobject3d')
    fixture_dataset(gso, 'gso')
    values = {**FAKE_OVERRIDES, 'data.source':'omniobject3d', 'data.frames':4,
              'paths.data_root':omni, 'paths.output_root':tmp_path/'out',
              'paths.scratch_root':tmp_path/'scratch', 'paths.cache_root':tmp_path/'cache',
              'train.max_steps':2, 'train.ckpt_every':1, 'adapter.pooler_queries':4, 'adapter.append_queries':4,
              'adapter.foreground_threshold':.01, 'logging.media_samples':0}
    base = load_config('c3', overrides=[f'{key}={value}' for key,value in values.items()])
    cfg = variant_config(base, variant, variant.name)
    cache_latents(cfg, 'train', build_sam3d(cfg, torch.device('cpu')), torch.device('cpu'))
    assert train(cfg, stop_at=1)['step'] == 1
    assert train(cfg)['step'] == 2
    run = tmp_path/'out'/variant.name
    state = torch.load(run/'ckpt_2.pt', weights_only=False)
    assert bool([name for name in state['extra'] if name.startswith('sam.')]) == variant.sam_lora
    assert bool([name for name in state['extra'] if name.startswith('encoder.')]) == variant.video_lora
    assert state['extra'].keys() == state['extra_ema'].keys()
    assert all(torch.isfinite(value).all() for value in state['adapter'].values())
    eval_cfg = replace_config(cfg, **{'data.source':'gso', 'data.eval_split':'eval',
                                      'paths.data_root':str(gso), 'eval.k':4})
    loaded, adapter, _ = load_run(eval_cfg, str(run), torch.device('cpu'))
    assert loaded.data.source == 'gso' and adapter is not None
    assert adapter.extra_ema_state.keys() == state['extra_ema'].keys()
    result = evaluate(eval_cfg, str(run))
    rows = read_jsonl(result)
    assert len(rows) == 1 and {'viou', 'viou_aligned', 'chamfer', 'fscore@0.05'} <= rows[0].keys()
    assert len(list((result.parent/'predictions'/variant.name).glob('*.pt'))) == 1
    if variant.name == 'fusion-frozen':
        baseline = read_jsonl(evaluate(eval_cfg, 'zero-shot'))
        assert len(baseline) == 1 and baseline[0]['condition'] == 'c0'
    assert json.loads((run/'config.json').read_text())['encoder']['kv_source'] == variant.encoder
    assert config_dict(cfg)['adapter']['strategy'] == variant.strategy


def test_resolve_full_matrix_and_foreground_launcher(tmp_path: Path) -> None:
    omni, gso = tmp_path/'omni', tmp_path/'gso'
    fixture_dataset(omni, 'omniobject3d')
    fixture_dataset(gso, 'gso')
    omni_rows = [row for row in records(omni) if row.split in {'train', 'val'}]
    for row in omni_rows:
        row.frames *= 2
        row.masks *= 2
    write_manifest(omni, omni_rows)
    # The resolver reads the existing preparation audit; tiny fixture assets are shared.
    (omni/'audit.json').write_text(json.dumps({'objects':2}))
    row = next(row for row in records(gso) if row.split == 'eval')
    row.frames *= 2
    row.masks *= 2
    write_manifest(gso, [row.model_copy(update={'object_id':f'object-{i}'}) for i in range(250)])
    (gso/'audit.json').write_text(json.dumps({'objects':250}))
    base = tmp_path/'base.json'
    base.write_text(json.dumps(config_dict(load_config('c3'))))
    root = tmp_path/'resolved'
    manifest = Path(configure(str(base), str(root), str(omni), str(gso), expected_objects=2))
    matrix = json.loads(manifest.read_text())
    assert matrix['train_objects'] == matrix['validation_objects'] == 1
    assert len(matrix['entries']) == 9
    for entry in matrix['entries']:
        cfg = json.loads(Path(entry['config']).read_text())
        assert cfg['data']['frames'] == 16 and cfg['encoder']['kv_source'] in {'vjepa','dinov2_frames'}
        assert cfg['train']['epochs'] == 10 and cfg['logging']['wandb']
    with pytest.raises(ValueError, match='immutable'):
        configure(str(base), str(root), str(omni), str(gso), expected_objects=2)
    # Exercise shell dispatch and its persisted exit marker through the actual CLI.
    # A tiny native baseline avoids redundant training already tested above.
    values = {**FAKE_OVERRIDES, 'data.source':'gso', 'data.eval_split':'eval',
              'paths.data_root':root/'data/gso', 'paths.output_root':root/'results',
              'paths.scratch_root':root/'scratch', 'paths.cache_root':root/'cache',
              'data.frames':4, 'eval.k':4, 'logging.media_samples':0, 'logging.wandb':'false'}
    cfg = load_config('c3', overrides=[f'{key}={value}' for key,value in values.items()])
    # Limit this fixture to one object after resolving the production-size selection.
    write_manifest(root/'data/gso', [row])
    (root/'configs/sam3d-baseline-eval.json').write_text(json.dumps(config_dict(cfg)))
    import sys
    script = Path(__file__).parents[1]/'experiments/conditioning_ablation/run.sh'
    subprocess.run(['bash', str(script), str(root), 'sam3d-baseline', sys.executable], check=True)
    assert json.loads((root/'logs/sam3d-baseline-exit.json').read_text())['exit_code'] == 0


@pytest.mark.parametrize('only', ['', 'profile-test'])
def test_batch_profiling_preserves_effective_batch_and_model(tmp_path: Path, only: str) -> None:
    from experiments.conditioning_ablation.profile import profile
    from tests.fixtures.fake_config import fake_train_cfg
    from tests.fixtures.make_fake_dataset import make_fake_dataset
    from video3d.config import read_config
    data = make_fake_dataset(tmp_path/'data')
    cfg = fake_train_cfg(data, tmp_path, condition='sam_video_lora')
    train_path, eval_path = tmp_path/'train.json', tmp_path/'eval.json'
    for path in [train_path, eval_path]:
        path.write_text(json.dumps(config_dict(cfg)))
    (tmp_path/'matrix.json').write_text(json.dumps({'entries':[{
        'name':'profile-test', 'baseline':False, 'config':str(train_path), 'evaluation':str(eval_path)}]}))
    profile(str(tmp_path), batches=(1,2), steps=2, only=only)
    ready = tmp_path/'profiles/profile-test-ready.json' if only else tmp_path/'profiles-ready.json'
    report = json.loads(ready.read_text())['profile-test']
    if only:
        assert (tmp_path/'profiles/profile-test-progress.json').exists()
        assert not (tmp_path/'profiles/progress.json').exists()
    assert len(report['candidates']) == 2
    assert report['selected']['batch_size'] in [1,2]
    resolved = read_config(train_path)
    assert resolved.train.effective_batch_size == cfg.train.effective_batch_size == 2
    assert resolved.sam3d == cfg.sam3d and resolved.encoder == cfg.encoder
    assert resolved.train.batch_size == read_config(eval_path).train.batch_size
