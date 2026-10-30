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
from video3d.data.scan_prepare import write_manifest
from video3d.data.samples import ClipDataset, split_objects
from video3d.data.reader import open_split
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


def synthetic_snapshot(source: Path, root: Path, base: ExperimentConfig,
                       expected: int) -> dict[str, Any]:
    """Reuse the export reader and its asset-disjoint split without copying shards."""
    destination = root/'data/synthetic'
    destination.mkdir(parents=True, exist_ok=True)
    (destination/base.data.train_split).symlink_to(source.resolve(), target_is_directory=True)
    split = open_split(destination/base.data.train_split, root/'cache')
    if len(split.clips) < expected or split.index['num_clips'] != len(split.clips):
        raise ValueError(f'Full synthetic export required: {len(split.clips)} clips, expected {expected}')
    if not split.voxels_file.is_file():
        raise ValueError('Synthetic canonical voxel targets are missing')
    assets = {cid: ref.asset_ids for cid, ref in split.clips.items()}
    train = split_objects(assets, base.data.train_split, base.data)
    val = split_objects(assets, 'val', base.data)
    train_assets = {assets[cid][k-1] for cid, ks in train.items() for k in ks}
    val_assets = {assets[cid][k-1] for cid, ks in val.items() for k in ks}
    if not train_assets or not val_assets or train_assets & val_assets or train.keys() & val.keys():
        raise ValueError('Nonempty object-disjoint synthetic train/val partitions are required')
    cfg = replace_config(base, **{'paths.data_root':str(destination),
                                  'paths.cache_root':str(root/'cache'), 'data.source':'synthetic'})
    train_ds, val_ds = ClipDataset(cfg, base.data.train_split, True), ClipDataset(cfg, 'val', False)
    report = {'source':str(source.resolve()), 'clips':len(split.clips),
              'train_clips':len(train), 'validation_clips':len(val),
              'train_assets':sorted(train_assets), 'validation_assets':sorted(val_assets),
              'train_objects':len(train_assets), 'validation_objects':len(val_assets),
              'train_samples':len(train_ds), 'validation_samples':len(val_ds),
              'val_asset_frac':base.data.val_asset_frac,
              'split_rule':'existing CRC32 val:asset_id; mixed clips excluded from training',
              'train_targets':train, 'validation_targets':val}
    atomic_json(destination/'split.json', report)
    # Freeze the source index in the run while retaining one physical shard copy.
    (destination/'source-index.json').write_bytes((source/'index.json').read_bytes())
    return report


def configure(base_config: str, output: str, omni_root: str = '', gso_root: str = '',
              expected_objects: int = 6000, epochs: int = 10, seed: int = 42,
              microbatch: int = 8, synthetic_root: str = '', expected_clips: int = 10000,
              synthetic_val_asset_frac: float = .03, evaluation_objects: int = 250) -> str:
    if microbatch not in [1, 2, 4, 8, 16, 32] or epochs < 1:
        raise ValueError('Positive epochs and a microbatch dividing effective batch 32 are required')
    if not 1 <= evaluation_objects <= 250:
        raise ValueError('evaluation_objects must be between 1 and 250')
    root = Path(output).absolute()
    if (root/'matrix.json').exists():
        raise ValueError('Choose a new experiment root; existing resolved configurations are immutable')
    if bool(omni_root) == bool(synthetic_root):
        raise ValueError('Specify exactly one of omni_root or synthetic_root')
    original = read_config(base_config)
    data_report: dict[str, Any]
    if synthetic_root:
        if not 0 < synthetic_val_asset_frac < 1:
            raise ValueError('Synthetic validation asset fraction must be between zero and one')
        original = replace_config(original, **{'data.val_asset_frac':synthetic_val_asset_frac,
                                               'data.clip_lengths':[16], 'data.clip_length_probs':[1.]})
        data_report = synthetic_snapshot(Path(synthetic_root), root, original, expected_clips)
        train_count, val_count = data_report['train_samples'], data_report['validation_samples']
        source, data_root = 'synthetic', root/'data/synthetic'
    else:
        audited_snapshot(Path(omni_root), root/'data/omni', expected_objects)
        omni_rows = records(Path(omni_root))
        if any(row.dataset != 'omniobject3d' or row.split not in {'train', 'val'} or len(row.frames) < 16 for row in omni_rows):
            raise ValueError('Omni requires train/val objects with at least 16 frames each')
        train_count = sum(row.split == 'train' for row in omni_rows)
        val_count = sum(row.split == 'val' for row in omni_rows)
        source, data_root = 'omniobject3d', root/'data/omni'
        data_report = {'train_objects':train_count, 'validation_objects':val_count,
                       'train_samples':train_count, 'validation_samples':val_count}
    gso_rows = records(Path(gso_root))
    if any(row.dataset != 'gso' or row.split != 'eval' or len(row.frames) < 16 for row in gso_rows):
        raise ValueError('GSO requires evaluation objects with at least 16 frames each')
    if len(gso_rows) != 250:
        raise ValueError('Use the fixed 250-object GSO evaluation selection')
    gso_audit = audited_snapshot(Path(gso_root), root/'data/gso', 250)
    selected = gso_rows[:evaluation_objects]
    write_manifest(root/'data/gso', selected)
    atomic_json(root/'data/gso/audit.json', {**gso_audit, 'objects':len(selected)})
    atomic_json(root/'data/gso/selection.json', {
        'source':str(Path(gso_root).resolve()), 'source_objects':250,
        'selection':'prefix of the existing frozen manifest',
        'object_ids':[row.object_id for row in selected]})
    if min(train_count, val_count) < 1:
        raise ValueError('Separate training and validation samples are required')
    steps = math.ceil(train_count/32)
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    base = replace_config(original, **{
        'seed':seed, 'env':'ucloud', 'device':'cuda', 'provenance.git_head':revision,
        'paths.data_root':str(data_root), 'paths.output_root':str(root/'results'),
        'paths.scratch_root':str(root/'scratch'), 'paths.cache_root':str(root/'cache'),
        'data.source':source, 'data.frames':16, 'data.eval_split':'val',
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
    atomic_json(root/'matrix.json', {'source_commit':revision, 'data_source':source,
        'train_objects':data_report['train_objects'], 'validation_objects':data_report['validation_objects'],
        'train_samples':train_count, 'validation_samples':val_count,
        'evaluation_objects':evaluation_objects,
        'epochs':epochs, 'steps_per_epoch':steps,
        'effective_batch':32, 'entries':entries,
        'protocol':'16 matched frames; existing native mask/pointmap conditioning; no DINO video encoder'})
    return str(root/'matrix.json')


if __name__ == '__main__':
    fire.Fire(configure)
