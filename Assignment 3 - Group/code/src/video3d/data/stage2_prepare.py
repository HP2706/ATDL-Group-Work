"""Canonical textured source meshes -> multiview DINO features for the existing SLAT cache.

Feature projection follows Microsoft's MIT-licensed TRELLIS extract_feature.py:
https://raw.githubusercontent.com/microsoft/TRELLIS/main/dataset_toolkits/extract_feature.py
Uses 518px DINOv2-L/14-register prenorm patch tokens, bilinear projection onto
surface voxel centers and averaging across all views (including occluded views).
Rendering uses our existing Blender normalizer and thumbnail renderer.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from video3d.config import ExperimentConfig
from video3d.data.stage2 import file_hash, validate_target
from video3d.storage import atomic_json, atomic_torch


def project_features(features: torch.Tensor, coords: torch.Tensor, cameras: dict[str, Any]) -> torch.Tensor:
    positions = (coords.to(features.device).float() + 0.5) / 64 - 0.5
    poses = torch.tensor(cameras['T_object_camera'], device=features.device, dtype=torch.float32)
    cv = poses.clone()
    cv[:, :3, 1:3] *= -1
    world_to_camera = torch.linalg.inv(cv)
    local = torch.einsum('vij,nj->vni', world_to_camera[:, :3, :3], positions) + world_to_camera[:, None, :3, 3]
    intrinsic = torch.tensor(cameras['K'], device=features.device, dtype=torch.float32)
    pixel = local @ intrinsic.T
    uv = (pixel[..., :2] / pixel[..., 2:]).div(cameras['resolution']) * 2 - 1
    # Pixel centers are integers; grid_sample's continuous image bounds lie half a pixel outside.
    uv += 1 / cameras['resolution']
    return F.grid_sample(features.float(), uv[:, None], align_corners=False).squeeze(2).mean(0).T


@torch.no_grad()
def prepare_features(cfg: ExperimentConfig, raw_root: str, output: str, asset_ids: list[str],
                     views: int = 32, batch_size: int = 4,
                     blender_install_path: str = '.blender',
                     blender_custom_path: str | None = None) -> dict[str, Any]:
    if sys.platform == 'darwin' or not torch.cuda.is_available():
        raise RuntimeError('Source-mesh feature preparation must run on a remote GPU')
    if views < 2 or batch_size < 1 or not asset_ids:
        raise ValueError('Specify assets, at least two views and a positive render-feature microbatch')
    if cfg.target.canonical_permutation != [[1, 0, 0], [0, 1, 0], [0, 0, 1]]:
        raise ValueError('Mesh recovery currently requires the original canonical axis convention')
    from video3d.data.generation.assets import objaverse_assets, preprocess
    from video3d.data.generation.blenderrun import run_blenderproc
    from video3d.data.generation.config import Config
    from video3d.data import build_dataset, ground_truth
    root = Path(output).resolve()
    root.mkdir(parents=True, exist_ok=True)
    raw = Path(raw_root)
    assets = {asset.asset_id: asset for asset in objaverse_assets(raw, None)}
    selected = [assets[uid] for uid in dict.fromkeys(asset_ids)]
    for asset in selected:
        if file_hash(raw / 'glbs' / f'{asset.asset_id}.glb') != asset.meta['sha256']:
            raise ValueError(f'Original source hash mismatch: {asset.asset_id}')
    generation = Config(dataset_root=str(root / 'canonical'), blender_install_path=blender_install_path,
                        blender_custom_path=blender_custom_path)
    preprocess(generation, selected, 'train', chunk=1, workers=1, retry_failed=True)
    missing = [asset.asset_id for asset in selected
               if not (root / 'canonical/assets/train' / asset.asset_id / 'meta.json').is_file()]
    if missing:
        raise RuntimeError(f'Canonical preprocessing failed for {missing}; inspect canonical/assets/failed_train.jsonl')
    recipe = {'version': 'source-mesh-dinov2-l14-reg-prenorm-v1', 'views': views,
              'view_distribution': 'fibonacci_sphere', 'resolution': 518, 'cycles_samples': 8,
              'feature_model': 'dinov2_vitl14_reg', 'pooling': 'all-view-mean-bilinear',
              'canonical_permutation': cfg.target.canonical_permutation,
              'renderer_sha256': file_hash(Path(__file__).parent / 'generation/blender/render_thumbs.py'),
              'normalizer_sha256': file_hash(Path(__file__).parent / 'generation/blender/normalize_assets.py')}
    manifest_path = root / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {
        'target_kind': 'source_mesh_sparse_features', 'feature_recipe': recipe, 'assets': {}}
    # Use the held-out split rules from the existing dataset; feature encoding is not optimization.
    ds = build_dataset(cfg, cfg.data.train_split, train=False)
    dino = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14_reg', trust_repo=True).cuda().eval()
    dino.requires_grad_(False)
    recipe['dino_weights_sha256'] = file_hash(Path(torch.hub.get_dir()) / 'checkpoints/dinov2_vitl14_reg4_pretrain.pth')
    # The stored identity includes frozen feature weights as well as render settings.
    if manifest_path.exists() and manifest['feature_recipe'] != recipe:
        raise ValueError('Feature weights changed; use a new directory')
    manifest['feature_recipe'] = recipe
    for asset in selected:
        uid = asset.asset_id
        if uid in manifest['assets']:
            entry = manifest['assets'][uid]
            if file_hash(root / entry['file']) != entry['sha256']:
                raise ValueError(f'Existing feature hash mismatch: {uid}')
            continue
        canonical = root / 'canonical/assets/train' / uid
        voxels = np.load(canonical / 'voxels.npy')
        gt = ground_truth(ds, uid, cfg)
        intersection, union = (voxels & gt).sum(), (voxels | gt).sum()
        agreement = float(intersection / max(union, 1))
        if agreement < .98:
            raise ValueError(f'Canonical recovery does not match dataset voxels: {uid}, IoU={agreement:.4f}')
        rgba_path, camera_path = canonical / 'feature-renders.npy', canonical / 'feature-cameras.json'
        job = {'mesh': str(canonical / 'mesh.glb'), 'out': str(rgba_path), 'cameras': str(camera_path),
               'views': views, 'resolution': 518, 'samples': 8}
        if not rgba_path.exists() or not camera_path.exists():
            jobs = root / 'render-jobs.json'
            atomic_json(jobs, [job])
            run_blenderproc('render_thumbs.py', str(jobs), install_path=blender_install_path,
                            custom_path=blender_custom_path, log_path=canonical / 'feature-render.log')
        rgba = np.load(rgba_path)
        cameras = json.loads(camera_path.read_text())
        coords = torch.from_numpy(np.argwhere(voxels).astype(np.int32))
        pooled = torch.zeros(len(coords), 1024, device='cuda')
        for start in range(0, len(rgba), batch_size):
            images = torch.from_numpy(rgba[start:start+batch_size].copy()).cuda().float() / 255
            images = (images[..., :3] * images[..., 3:]).permute(0, 3, 1, 2)
            mean = images.new_tensor([.485, .456, .406])[None, :, None, None]
            std = images.new_tensor([.229, .224, .225])[None, :, None, None]
            with torch.autocast('cuda', dtype=torch.float16):
                encoded = dino((images-mean)/std, is_training=True)['x_prenorm'][:, 5:]
            patches = encoded.transpose(1, 2).reshape(len(images), 1024, 37, 37)
            batch_cameras = {**cameras, 'T_object_camera': cameras['T_object_camera'][start:start+len(images)]}
            pooled += project_features(patches, coords, batch_cameras) * len(images) / views
        target = validate_target(coords, pooled.cpu(), channels=1024)
        destination = root / f'{uid}.pt'
        if destination.exists():
            raise FileExistsError(f'Unjournaled features preserved: {destination}')
        atomic_torch(destination, {'coords': target.coords, 'features': target.features})
        manifest['assets'][uid] = {'file': destination.name, 'sha256': file_hash(destination),
            'source_mesh_sha256': file_hash(raw / 'glbs' / f'{uid}.glb'),
            'normalization_sha256': file_hash(canonical / 'normalize.json'),
            'mesh_path': str(canonical / 'mesh.ply'), 'mesh_sha256': file_hash(canonical / 'mesh.ply'),
            'voxel_agreement_iou': agreement}
        atomic_json(manifest_path, manifest)
    return {'assets': len(manifest['assets']), 'manifest': str(manifest_path)}
