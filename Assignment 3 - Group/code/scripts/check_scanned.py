"""Check real prepared inputs through each model configuration's CPU smoke model.

This does not download model weights or validate the pretrained GPU models.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import fire
import torch
from torch.utils.data import DataLoader

from video3d.config import load_config
from video3d.data import build_dataset
from video3d.data.scanned import audit, records
from video3d.models.adapter import build_adapter
from video3d.models.encoders import build_kv_encoder
from video3d.models.sam3d import build_sam3d
from video3d.train import batch_loss, collate_list


def main(root: str, source: str, report: str = '') -> dict[str, Any]:
    torch.set_num_threads(2)
    rows = records(Path(root))
    audit_report = audit(root)
    splits = sorted({row.split for row in rows})
    checks: list[dict[str, Any]] = []
    for condition in ['c3', 'sam_lora', 'sam_video_lora', 'dino_pooler', 'dino_pooler_single']:
        values = {'fake_sam3d': True, 'device': 'cpu', 'data.source': source,
                  'paths.data_root': root, 'data.frames': 16, 'data.crop.size': 32,
                  'data.crop.min_side': 16, 'adapter.d_cond': 32, 'adapter.d_vid': 32,
                  'adapter.heads': 4, 'adapter.query_sets': '[crop,full]',
                  'adapter.foreground_threshold': .01, 'train.bf16': False}
        cfg = load_config(condition, overrides=[f'{key}={value}' for key, value in values.items()])
        sam = build_sam3d(cfg, torch.device('cpu'))
        encoder = build_kv_encoder(cfg, sam, torch.device('cpu'))
        adapter = build_adapter(cfg.adapter)
        for split in splits:
            dataset = build_dataset(cfg, split, train=split == 'train')
            loader = DataLoader(dataset, batch_size=min(2, len(dataset)), num_workers=2, collate_fn=collate_list)
            batch = next(iter(loader))
            for sample in batch:
                voxels = torch.from_numpy(dataset.target_voxels(sample['info']['asset_id']))
                sample['target'] = sam.encode_voxels(voxels[None])[0].detach()
                if sample['inputs']['frames'].shape != (16, 3, 32, 32):
                    raise ValueError('Unexpected video shape')
                if not torch.isfinite(sample['inputs']['frames']).all():
                    raise ValueError('Nonfinite frame tensor')
            loss = batch_loss(sam, encoder, adapter, batch, cfg, torch.device('cpu'),
                              torch.Generator().manual_seed(42))
            loss.backward()
            gradients = [p.grad for p in adapter.parameters() if p.grad is not None]
            if not torch.isfinite(loss) or not gradients or not all(torch.isfinite(g).all() for g in gradients):
                raise ValueError(f'Invalid loss/gradients: {condition}/{split}')
            adapter.zero_grad(set_to_none=True)
            checks.append({'condition': condition, 'split': split, 'objects': len(dataset),
                           'loss': float(loss.detach()), 'finite_backward': True,
                           'sample_objects': [s['info']['asset_id'] for s in batch]})
    result = {'source': source, 'audit': audit_report, 'checks': checks,
              'model_validation': 'CPU smoke models only; pretrained GPU validation pending',
              'geometry_registration': 'rendered cameras' if source == 'gso' else 'scan/video registration not verified'}
    path = Path(report) if report else Path(root) / 'loader-check.json'
    path.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
    return result


if __name__ == '__main__':
    fire.Fire(main)
