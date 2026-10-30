"""SAM-only control reuses training, checkpointing, and evaluation."""
from __future__ import annotations

import json
from pathlib import Path

import torch

from experiments.conditioning_ablation.configure import Variant, variant_config
from tests.fixtures.fake_config import fake_train_cfg
from tests.fixtures.make_fake_dataset import make_fake_dataset
from video3d.config import replace_config
from video3d.data.samples import ClipDataset
from video3d.evaluate import evaluate
from video3d.models.adapter import build_adapter
from video3d.models.encoders import build_kv_encoder
from video3d.models.sam3d import build_sam3d
from video3d.train import adapted_condition, reference_images, train
from video3d.utils import read_jsonl


def test_native_random_frame_train_resume_and_evaluate(tmp_path: Path) -> None:
    data = make_fake_dataset(tmp_path/'data', size=32)
    base = fake_train_cfg(data, tmp_path, **{'train.max_steps':2, 'train.ckpt_every':1,
                                           'train.log_every':1, 'logging.media_samples':0})
    variant = Variant(name='sam3d-random-frame-lora', strategy='native', sam_lora=True, video_lora=False)
    cfg = variant_config(base, variant, variant.name)
    ds = ClipDataset(cfg, 'train', True)
    draws = [ds[(0, epoch)] for epoch in range(12)]
    assert len({s['info']['r'] for s in draws}) > 1
    assert all(s['inputs']['frames'].shape[0] == 1 for s in draws)
    assert ds[(0, 3)]['info']['r'] == ds[(0, 3)]['info']['r']
    sam = build_sam3d(cfg, torch.device('cpu'))
    encoder = build_kv_encoder(cfg, sam, torch.device('cpu'))
    adapter = build_adapter(cfg.adapter)
    assert sum(p.numel() for p in encoder.parameters()) == 0
    assert sum(p.numel() for p in adapter.parameters()) == 0
    rgb, mask = reference_images(draws[:1], torch.device('cpu'))
    expected = sam.condition(rgb, mask)
    # No video tensors are needed by native conditioning.
    batch = [{'inputs':{'ref_rgb':draws[0]['inputs']['ref_rgb'], 'ref_mask':draws[0]['inputs']['ref_mask']}}]
    actual = adapted_condition(sam, encoder, adapter, batch, cfg, torch.device('cpu'))
    assert all(torch.equal(actual[key], expected[key]) for key in expected)
    assert train(cfg, stop_at=1)['step'] == 1
    assert train(cfg)['step'] == 2
    run = tmp_path/'out'/variant.name
    first = torch.load(run/'ckpt_1.pt', weights_only=False)
    second = torch.load(run/'ckpt_2.pt', weights_only=False)
    assert not second['adapter'] and not second['ema']
    assert second['extra'] and all(key.startswith('sam.') for key in second['extra'])
    assert any(not torch.equal(first['extra'][key], second['extra'][key]) for key in second['extra'])
    counts = json.loads((run/'parameter_counts.json').read_text())
    assert counts['adapter'] == counts['video_total'] == counts['video_lora'] == 0
    assert counts['sam_lora'] == counts['trainable_total'] > 0
    evaluated = evaluate(replace_config(cfg, **{'data.eval_split':'val'}), str(run), limit=1)
    assert len(read_jsonl(evaluated)) == 1
