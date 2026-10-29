from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from tests.fixtures.fake_config import FAKE_OVERRIDES
from video3d.config import load_config, replace_config
from video3d.data import build_dataset
from video3d.data.scan_prepare import write_manifest
from video3d.data.scanned import ScanRecord, audit
from video3d.data.targets import cache_latents, latent_path
from video3d.models.encoders import FakeEncoder, kv_tokens
from video3d.models.sam3d import build_sam3d
from video3d.train import train


def fixture_dataset(root: Path, source: str) -> None:
    root.mkdir()
    image = np.full((48, 64, 3), 120, dtype=np.uint8)
    mask = np.zeros((48, 64), dtype=np.uint8)
    mask[12:36, 16:48] = 255
    Image.fromarray(image).save(root / 'rgb.jpg')
    Image.fromarray(mask).save(root / 'mask.jpg')
    voxels = np.zeros((64, 64, 64), dtype=bool)
    voxels[24:40, 24:40, 24:40] = True
    np.save(root / 'voxels.npy', voxels)
    (root / 'mesh.glb').write_bytes(b'fixture-only')
    rows = [ScanRecord(dataset=source, object_id=split, category='fixture', split=split,
                       frames=['rgb.jpg'] * 8, masks=['mask.jpg'] * 8,
                       mesh='mesh.glb', voxels='voxels.npy') for split in ['train', 'val', 'eval']]
    write_manifest(root, rows)


@pytest.mark.parametrize('source', ['omniobject3d', 'gso'])
@pytest.mark.parametrize('condition', ['c3', 'video_lora', 'sam_lora', 'sam_video_lora', 'dino_pooler', 'dino_pooler_single'])
def test_scanned_shared_trainer_resume(tmp_path: Path, source: str, condition: str) -> None:
    root = tmp_path / 'data'
    fixture_dataset(root, source)
    values = {**FAKE_OVERRIDES, 'data.source': source, 'data.frames': 4,
              'paths.data_root': root, 'paths.output_root': tmp_path / 'out',
              'paths.scratch_root': tmp_path / 'scratch', 'paths.cache_root': tmp_path / 'cache',
              'run_name': 'integration', 'train.max_steps': 2, 'train.ckpt_every': 1,
              'adapter.foreground_threshold': 0.01}
    cfg = load_config(condition, overrides=[f'{k}={v}' for k, v in values.items()])
    dataset = build_dataset(cfg, 'train', True)
    sample = dataset[0]
    assert sample['inputs']['frames'].shape == (4, 3, 32, 32)
    assert sample['inputs']['ref_rgb'].shape == (512, 512, 3)
    assert sample['inputs']['ref_mask'].shape == (512, 512)
    assert len(sample['info']['frame_ids']) == len(set(sample['info']['frame_ids'])) == 4
    assert dataset.target_voxels('train').shape == (64, 64, 64)
    report = audit(str(root))
    assert report['objects'] == 3 and report['frames'] == 24
    model = build_sam3d(cfg, torch.device('cpu'))
    for split in ['train', 'val']:
        cache_latents(cfg, split, model, torch.device('cpu'))
    assert latent_path(cfg, 'train').is_file()
    assert train(cfg, stop_at=1)['step'] == 1
    assert train(cfg)['step'] == 2
    state = torch.load(tmp_path / 'out/integration/ckpt_2.pt', weights_only=False)
    assert state['step'] == 2
    assert all(torch.isfinite(tensor).all() for tensor in state['adapter'].values())
    if source == 'omniobject3d' and condition == 'c3':
        from video3d.evaluate import load_run
        gso_root = tmp_path / 'gso'
        fixture_dataset(gso_root, 'gso')
        eval_cfg = replace_config(cfg, **{'data.source': 'gso', 'paths.data_root': str(gso_root)})
        loaded, _, _ = load_run(eval_cfg, str(tmp_path / 'out/integration'), torch.device('cpu'))
        assert loaded.data.source == 'gso'
        assert loaded.paths.data_root == str(gso_root)
        assert len(build_dataset(loaded, 'eval', False)) == 1


def test_dino_frame_tokens_are_not_pair_averaged() -> None:
    encoder = FakeEncoder(8, 16, patch=16, tubelet=1)
    frames = torch.stack([torch.full((3, 16, 16), float(i)) for i in range(4)])
    masks = torch.tensor([1., 0., 0., 1.]).view(4, 1, 1).expand(4, 16, 16)
    inputs = {'frames': frames, 'frame_masks': masks, 'dt': torch.tensor([0, 1]), 'ref_pos': 0}
    tokens, coverage, dt = kv_tokens(encoder, inputs, 'dinov2_frames')
    assert tokens.shape == (4, 8)
    assert not torch.equal(tokens[0], tokens[1])
    assert coverage.tolist() == [1., 0., 0., 1.]
    assert dt.tolist() == [0, 1, 2, 3]


def test_manifest_allows_owned_capture_symlinks_but_rejects_traversal(tmp_path: Path) -> None:
    from video3d.data.scanned import records
    root = tmp_path / 'prepared'
    fixture_dataset(root, 'omniobject3d')
    external = tmp_path / 'raw'
    external.mkdir()
    (root / 'captures').symlink_to(external, target_is_directory=True)
    rows = records(root)
    rows[0].frames = ['captures/frame.jpg'] * 8
    write_manifest(root, rows)
    assert len(records(root)) == 3
    rows[0].frames = ['../escape.jpg'] * 8
    write_manifest(root, rows)
    with pytest.raises(ValueError, match='Unsafe manifest'):
        records(root)
