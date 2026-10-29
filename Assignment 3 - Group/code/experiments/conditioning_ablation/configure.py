"""Resolve one matched experiment matrix through the existing chz configuration API."""
from __future__ import annotations

from datetime import datetime
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Literal

import fire
from pydantic import BaseModel, ConfigDict

from video3d.config import ExperimentConfig, config_dict, read_config, replace_config
from video3d.data.scanned import records
from video3d.storage import atomic_json


class Variant(BaseModel):
    model_config = ConfigDict(extra='forbid')
    name: str
    strategy: Literal["fusion", "pooler"]
    sam_lora: bool
    video_lora: bool
    baseline: bool = False


def variants() -> list[Variant]:
    return [Variant.model_validate(row) for row in json.loads(Path(__file__).with_name('variants.json').read_text())]


def variant_config(base: ExperimentConfig, variant: Variant, run_name: str) -> ExperimentConfig:
    """Only these three model switches differ; all image/video preprocessing stays matched."""
    return replace_config(base, **{'run_name': run_name, 'condition': variant.name,
        'adapter.strategy': variant.strategy, 'encoder.kv_source': 'vjepa',
        'sam3d.lora': variant.sam_lora, 'encoder.vjepa.video_lora': variant.video_lora})


def audited_snapshot(source: Path, destination: Path, expected: int) -> dict[str, Any]:
    report = json.loads((source/'audit.json').read_text())
    rows = records(source)
    if len(rows) < expected or report['objects'] != len(rows):
        raise ValueError(f'Full audited dataset required: {source} has {len(rows)}, expected at least {expected}')
    destination.mkdir(parents=True, exist_ok=True)
    manifest = source/'manifest.jsonl'
    target = destination/'manifest.jsonl'
    if target.exists() and target.read_bytes() != manifest.read_bytes():
        raise ValueError('Immutable experiment manifest already differs')
    target.write_bytes(manifest.read_bytes())
    # Keep one physical copy of generated assets; never alter the preparation task's files.
    for path in source.iterdir():
        if path.name not in {'manifest.jsonl', 'audit.json'} and not (destination/path.name).exists():
            (destination/path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    atomic_json(destination/'audit.json', report)
    return report


def configure(base_config: str, output: str, omni_root: str, gso_root: str,
              expected_objects: int = 6000, epochs: int = 10, seed: int = 42,
              microbatch: int = 8) -> str:
    if microbatch not in [1, 2, 4, 8, 16, 32] or epochs < 1:
        raise ValueError('Positive epochs and a microbatch dividing effective batch 32 are required')
    root = Path(output).absolute()
    if (root/'matrix.json').exists():
        raise ValueError('Choose a new experiment root; existing resolved configurations are immutable')
    audited_snapshot(Path(omni_root), root/'data/omni', expected_objects)
    omni_rows = records(Path(omni_root))
    gso_rows = records(Path(gso_root))
    if any(row.dataset != 'omniobject3d' or row.split not in {'train', 'val'} or len(row.frames) < 16 for row in omni_rows):
        raise ValueError('Omni requires train/val objects with at least 16 frames each')
    if any(row.dataset != 'gso' or row.split != 'eval' or len(row.frames) < 16 for row in gso_rows):
        raise ValueError('GSO requires evaluation objects with at least 16 frames each')
    if len(gso_rows) != 250:
        raise ValueError('Use the fixed 250-object GSO evaluation selection')
    audited_snapshot(Path(gso_root), root/'data/gso', 250)
    rows = records(root/'data/omni')
    train_count = sum(row.split == 'train' for row in rows)
    val_count = sum(row.split == 'val' for row in rows)
    if min(train_count, val_count) < 1:
        raise ValueError('Separate Omni training and validation objects are required')
    steps = math.ceil(train_count/32)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    base = replace_config(read_config(base_config), **{
        'seed':seed, 'env':'ucloud', 'device':'cuda', 'provenance.git_head':revision,
        'paths.data_root':str(root/'data/omni'), 'paths.output_root':str(root/'results'),
        'paths.scratch_root':str(root/'scratch'), 'paths.cache_root':str(root/'cache'),
        'data.source':'omniobject3d', 'data.frames':16, 'data.eval_split':'val',
        'data.num_workers':4, 'train.epochs':epochs, 'train.batch_size':microbatch,
        'train.effective_batch_size':32, 'train.max_seconds':36000,
        'train.ckpt_every':50, 'train.log_every':1,
        'train.resume':'auto', 'val.epoch_media_samples':0,
        'val.loss_every':steps, 'val.loss_num_samples':min(50,val_count),
        'val.sample_every':steps, 'val.sample_num_objects':min(50,val_count),
        'val.tau_grid':[.1,.5,.9], 'optim.lr':1e-4, 'optim.warmup_steps':max(1,steps*epochs//20),
        'optim.ema_decay':.99, 'logging.wandb':True, 'eval.k':16})
    entries = []
    for variant in variants():
        cfg = variant_config(base, variant, f'{variant.name}-seed{seed}-{stamp}')
        train_path = root/'configs'/f'{variant.name}.json'
        atomic_json(train_path, config_dict(cfg))
        evaluation = replace_config(cfg, **{'data.source':'gso', 'data.eval_split':'eval',
            'paths.data_root':str(root/'data/gso'), 'sampler.mode':'shortcut', 'sampler.steps':4})
        eval_path = root/'configs'/f'{variant.name}-eval.json'
        atomic_json(eval_path, config_dict(evaluation))
        entries.append({**variant.model_dump(), 'config':str(train_path), 'evaluation':str(eval_path)})
    atomic_json(root/'matrix.json', {'source_commit':revision, 'train_objects':train_count,
        'validation_objects':val_count, 'epochs':epochs, 'steps_per_epoch':steps,
        'effective_batch':32, 'entries':entries,
        'protocol':'16 matched frames; existing native mask/pointmap conditioning; no DINO video encoder'})
    return str(root/'matrix.json')


if __name__ == '__main__':
    fire.Fire(configure)
