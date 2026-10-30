"""Verified per-asset structured latent targets; never infer targets from model predictions."""
from __future__ import annotations

import hashlib
import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict
import torch

from video3d.config import ExperimentConfig


class StructuredTarget(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    coords: torch.Tensor
    features: torch.Tensor


def validate_target(coords: torch.Tensor, features: torch.Tensor, channels: int = 8) -> StructuredTarget:
    if coords.ndim != 2 or coords.shape[1] != 3 or len(coords) == 0:
        raise ValueError('Stage-2 coordinates must be nonempty N x 3 canonical voxel indices')
    if coords.dtype not in (torch.int32, torch.int64) or bool(((coords < 0) | (coords >= 64)).any()):
        raise ValueError('Stage-2 coordinates must be integers in [0, 64)')
    if len(torch.unique(coords, dim=0)) != len(coords):
        raise ValueError('Duplicate Stage-2 coordinates')
    if features.shape != (len(coords), channels) or not bool(torch.isfinite(features).all()):
        raise ValueError(f'Stage-2 features must be finite N x {channels}')
    return StructuredTarget(coords=coords.int(), features=features.float())


def file_hash(path: Path) -> str:
    with path.open('rb') as handle:
        return hashlib.file_digest(handle, 'sha256').hexdigest()


def target_manifest(cfg: ExperimentConfig) -> dict[str, Any]:
    path = Path(cfg.stage2.target_root) / 'manifest.json'
    stat = path.stat()
    manifest = read_target_manifest(str(path.resolve()), stat.st_mtime_ns, stat.st_size)
    if (manifest['schema_version'] != 1 or manifest['target_kind'] != 'source_mesh_slat_encoder_mean'
            or manifest['sam_revision'] != cfg.sam3d.checkpoint_revision
            or manifest['coordinates'] != 'canonical-surface64-xyz'
            or manifest['features'] != 'unnormalized-slat-encoder-mean'):
        raise ValueError('Stage-2 target manifest is incompatible with this model')
    return manifest


@lru_cache(maxsize=8)
def read_target_manifest(path: str, mtime_ns: int, size: int) -> dict[str, Any]:
    """Parse an immutable manifest once, invalidate when a preparation journal changes."""
    return json.loads(Path(path).read_text())


def load_target(cfg: ExperimentConfig, asset_id: str) -> StructuredTarget:
    root = Path(cfg.stage2.target_root).resolve()
    entry = target_manifest(cfg)['assets'][asset_id]
    path = (root / entry['file']).resolve()
    if not path.is_relative_to(root):
        raise ValueError('Stage-2 target path escapes its cache root')
    payload = torch.load(path, map_location='cpu', weights_only=True)
    if payload['asset_id'] != asset_id or payload['source_mesh_sha256'] != entry['source_mesh_sha256']:
        raise ValueError('Stage-2 asset identity mismatch')
    return validate_target(payload['coords'], payload['features'])


def audit_targets(cfg: ExperimentConfig, asset_ids: list[str]) -> None:
    manifest = target_manifest(cfg)
    missing = sorted(set(asset_ids) - manifest['assets'].keys())
    if missing:
        raise FileNotFoundError(f'Missing {len(missing)} Stage-2 targets; first: {missing[:5]}')
    for asset_id in sorted(set(asset_ids)):
        entry = manifest['assets'][asset_id]
        path = Path(cfg.stage2.target_root) / entry['file']
        if file_hash(path) != entry['sha256']:
            raise ValueError(f'Stage-2 target hash mismatch: {asset_id}')
        load_target(cfg, asset_id)


@torch.no_grad()
def cache_targets(cfg: ExperimentConfig, features_root: str) -> dict[str, Any]:
    """Encode canonical source-mesh sparse DINO features, prepared by a verified renderer.

    Input manifest records the feature recipe, source geometry/normalization hashes,
    and per-asset N x 1024 surface features. A downloaded GLB alone is insufficient.
    This deliberately refuses teacher-generated Stage-2 predictions as ground truth.
    """
    if not cfg.stage2.target_root:
        raise ValueError('Set stage2.target_root to a new or compatible cache directory')
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    from video3d.storage import atomic_json, atomic_torch

    if not torch.cuda.is_available():
        raise RuntimeError('Stage-2 encoding requires remote GPU execution')
    source = Path(features_root).resolve()
    index = json.loads((source / 'manifest.json').read_text())
    if index['target_kind'] != 'source_mesh_sparse_features' or not index['feature_recipe']:
        raise ValueError('Expected documented source-mesh sparse features, not generated predictions')
    checkpoint = Path(cfg.paths.sam3d_checkpoint_dir)
    encoder = instantiate(OmegaConf.load(checkpoint / 'slat_encoder.yaml')).cuda().eval()
    weights = torch.load(checkpoint / 'slat_encoder.ckpt', map_location='cpu', weights_only=True)
    encoder.load_state_dict(weights, strict=True)
    encoder.requires_grad_(False)
    output = Path(cfg.stage2.target_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / 'manifest.json'
    manifest = {'schema_version': 1, 'target_kind': 'source_mesh_slat_encoder_mean',
                'sam_revision': cfg.sam3d.checkpoint_revision, 'coordinates': 'canonical-surface64-xyz',
                'features': 'unnormalized-slat-encoder-mean', 'feature_recipe': index['feature_recipe'],
                'source_manifest_sha256': file_hash(source / 'manifest.json'), 'assets': {}}
    if manifest_path.exists():
        previous = target_manifest(cfg)
        if previous['source_manifest_sha256'] != manifest['source_manifest_sha256']:
            raise ValueError('Source feature manifest changed; preserve the cache and choose a new directory')
        manifest = previous
    for asset_id, entry in index['assets'].items():
        if Path(asset_id).name != asset_id or asset_id in {'.', '..'}:
            raise ValueError('Stage-2 asset ID must be a safe filename')
        if asset_id in manifest['assets']:
            audit_targets(cfg, [asset_id])
            continue
        path = (source / entry['file']).resolve()
        if not path.is_relative_to(source) or file_hash(path) != entry['sha256']:
            raise ValueError(f'Invalid source feature file: {asset_id}')
        raw = torch.load(path, map_location='cpu', weights_only=True)
        target = validate_target(raw['coords'], raw['features'], channels=1024)
        coords = torch.cat([torch.zeros(len(target.coords), 1, dtype=torch.int32), target.coords], dim=1).cuda()
        with torch.autocast('cuda', dtype=torch.float16):
            encoded = encoder(sp.SparseTensor(coords=coords, feats=target.features.cuda()), sample_posterior=False)
        latent = validate_target(encoded.coords[:, 1:].cpu(), encoded.feats.float().cpu())
        destination = output / f'{asset_id}.pt'
        if destination.exists():
            raise FileExistsError(f'Unjournaled target preserved: {destination}')
        atomic_torch(destination, {'asset_id': asset_id, 'coords': latent.coords, 'features': latent.features,
                                  'source_mesh_sha256': entry['source_mesh_sha256'],
                                  'normalization_sha256': entry['normalization_sha256']})
        manifest['assets'][asset_id] = {'file': destination.name, 'sha256': file_hash(destination),
                                      'source_mesh_sha256': entry['source_mesh_sha256'],
                                      'normalization_sha256': entry['normalization_sha256'],
                                      **{key: entry[key] for key in ('mesh_path', 'mesh_sha256', 'voxel_agreement_iou') if key in entry}}
        atomic_json(manifest_path, manifest)
    return {'assets': len(manifest['assets']), 'manifest': str(manifest_path)}


def source_mesh(cfg: ExperimentConfig, asset_id: str) -> Path:
    entry = target_manifest(cfg)['assets'][asset_id]
    path = Path(entry['mesh_path'])
    if file_hash(path) != entry['mesh_sha256']:
        raise ValueError(f'Stage-2 source mesh hash mismatch: {asset_id}')
    return path
