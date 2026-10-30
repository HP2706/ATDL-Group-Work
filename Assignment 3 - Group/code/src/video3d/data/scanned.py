"""Real captures and rendered scans through the existing training sample contract."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import torch
from torch.utils.data import Dataset

from video3d.config import ExperimentConfig
from video3d.data.samples import check_no_leakage, preprocess_video, tubelet_offsets


class ScanRecord(BaseModel):
    dataset: Literal['omniobject3d', 'gso']
    object_id: str
    category: str
    split: str
    frames: list[str] = Field(min_length=1)
    masks: list[str] = Field(min_length=1)
    mesh: str
    voxels: str
    cameras: str | None = None


def records(root: Path) -> list[ScanRecord]:
    result = [ScanRecord.model_validate_json(line) for line in (root / 'manifest.jsonl').read_text().splitlines() if line.strip()]
    identities: set[str] = set()
    for row in result:
        if row.object_id in identities:
            raise ValueError(f'Duplicate object identity: {row.object_id}')
        identities.add(row.object_id)
        if len(row.frames) != len(row.masks):
            raise ValueError(f'Frame/mask count mismatch: {row.object_id}')
        for name in [*row.frames, *row.masks, row.mesh, row.voxels]:
            path = Path(name)
            if path.is_absolute() or '..' in path.parts:
                raise ValueError(f'Unsafe manifest path: {name}')
    return result


def object_split(object_id: str, category: str, seed: int, validation_fraction: float = 0.1) -> str:
    score = int.from_bytes(hashlib.sha256(f'{seed}:{category}:{object_id}'.encode()).digest()[:8], 'big') / 2**64
    return 'val' if score < validation_fraction else 'train'


def read_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.array(ImageOps.exif_transpose(image).convert('RGB'))


def read_mask(path: Path, threshold: float) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert('L')) >= round(255 * threshold)


class ScannedDataset(Dataset[dict[str, Any]]):
    """One item per object; deterministic epoch-dependent ordered video sampling."""
    def __init__(self, cfg: ExperimentConfig, split: str, train: bool, clip_len: int | None = None) -> None:
        self.cfg, self.train = cfg, train
        self.root = Path(cfg.paths.data_root)
        self.rows = [r for r in records(self.root) if r.split == split and r.dataset == cfg.data.source]
        if not self.rows:
            raise ValueError(f'No {cfg.data.source} objects in split {split}')
        self.clip_len = clip_len or cfg.data.frames
        if self.clip_len < 1 or (self.clip_len > 1 and self.clip_len % 2):
            raise ValueError('Clip length must be 1 or a positive even number')
        self.index = [{'clip_id': r.object_id, 'asset_id': r.object_id, 'k': 1, 'r': 0, 'rules': ['reference']} for r in self.rows]
        self.epoch = 0

    def __len__(self) -> int:
        return len(self.rows)

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def frame_ids(self, row: ScanRecord, rng: np.random.Generator | None) -> np.ndarray:
        count = len(row.frames)
        if count < self.clip_len:
            raise ValueError(f'{row.object_id} has {count} frames, needs {self.clip_len}')
        if rng is None:
            return np.linspace(0, count - 1, self.clip_len).round().astype(int)
        if self.clip_len == 1:
            return np.array([int(rng.integers(count))])
        strides = [s for s in self.cfg.data.scanned_strides if (self.clip_len - 1) * s < count]
        if not strides:
            raise ValueError(f'No valid temporal stride for {row.object_id}')
        stride = int(rng.choice(strides))
        start = int(rng.integers(count - (self.clip_len - 1) * stride))
        return start + stride * np.arange(self.clip_len)

    def __getitem__(self, position: int | tuple[int, int]) -> dict[str, Any]:
        index, epoch = position if isinstance(position, tuple) else (position, self.epoch)
        row = self.rows[index]
        rng = np.random.default_rng([self.cfg.seed, epoch, index]) if self.train else None
        ids = self.frame_ids(row, rng)
        images = [read_rgb(self.root / row.frames[t]) for t in ids]
        masks = [read_mask(self.root / row.masks[t], self.cfg.data.mask_threshold) for t in ids]
        for image, mask in zip(images, masks, strict=True):
            if image.shape[:2] != mask.shape or not mask.any():
                raise ValueError(f'Invalid RGB/mask pair in {row.object_id}')
        frames, frame_masks, _ = preprocess_video(images, masks, self.cfg.data, rng)
        reference = int(rng.integers(self.clip_len)) if rng is not None else 0
        ref_rgb = ImageOps.pad(Image.fromarray(images[reference]), (512, 512), color=(127, 127, 127))
        ref_mask = ImageOps.pad(Image.fromarray(masks[reference].astype(np.uint8) * 255), (512, 512), method=Image.Resampling.NEAREST, color=0)
        inputs = {'frames': frames, 'frame_masks': frame_masks,
                  'dt': torch.from_numpy(tubelet_offsets(self.clip_len, reference)), 'ref_pos': reference,
                  'ref_rgb': torch.from_numpy(np.array(ref_rgb)),
                  'ref_mask': torch.from_numpy(np.array(ref_mask) > 127)}
        check_no_leakage(inputs)
        return {'inputs': inputs, 'info': {**self.index[index], 'r': int(ids[reference]),
                'frame_ids': ids.tolist(), 'reference_rules': ['reference'], 'visible_frac': None,
                'amodal_px': None, 'arc_span_deg': None, 'num_objects': 1,
                'ground_truth_mesh': str(self.root / row.mesh), 'visibility': None,
                'dataset': row.dataset, 'category': row.category}}

    def target_voxels(self, asset_id: str) -> np.ndarray:
        from video3d.data.targets import apply_canonical_permutation, fill_voxels
        row = next(r for r in self.rows if r.object_id == asset_id)
        voxels = np.load(self.root / row.voxels, allow_pickle=False).astype(bool)
        if voxels.shape != (64, 64, 64) or not voxels.any():
            raise ValueError(f'Invalid occupancy for {asset_id}')
        voxels = apply_canonical_permutation(voxels, self.cfg.target.canonical_permutation)
        return fill_voxels(voxels) if self.cfg.target.voxel_fill == 'solid' else voxels


def audit(root: str, samples_per_object: int = 4) -> dict[str, Any]:
    base = Path(root)
    rows = records(base)
    counts: dict[str, int] = {}
    frames = 0
    for row in rows:
        counts[row.split] = counts.get(row.split, 0) + 1
        for name in [*row.frames, *row.masks, row.mesh, row.voxels]:
            if not (base / name).is_file() or (base / name).stat().st_size == 0:
                raise ValueError(f'Missing/empty file: {name}')
        voxels = np.load(base / row.voxels, allow_pickle=False)
        if voxels.shape != (64, 64, 64) or not voxels.any():
            raise ValueError(f'Invalid occupancy: {row.object_id}')
        for t in np.linspace(0, len(row.frames) - 1, min(samples_per_object, len(row.frames))).round().astype(int):
            image = read_rgb(base / row.frames[t])
            mask = read_mask(base / row.masks[t], 0.5)
            if image.shape[:2] != mask.shape or not mask.any():
                raise ValueError(f'Invalid image/mask: {row.object_id}/{t}')
        frames += len(row.frames)
    result = {'objects': len(rows), 'frames': frames, 'splits': counts,
              'file_existence_checked': 'all', 'decoded_frames_per_object': samples_per_object,
              'mesh_camera_registration': 'known_render_poses' if all(r.dataset == 'gso' for r in rows) else 'not_verified',
              'geometry_latent_cache': 'requires_explicit_cache_latents_stage'}
    (base / 'audit.json').write_text(json.dumps(result, indent=2))
    return result
