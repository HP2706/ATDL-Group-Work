"""Prepare existing scanned assets and captured frames without synthesizing training videos."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import trimesh

from video3d.data.scanned import ScanRecord, object_split


def normalize_mesh(source: Path, output: Path) -> tuple[trimesh.Trimesh, dict[str, Any]]:
    """Use the project's existing TRELLIS surface voxelizer for both scanned sources."""
    from video3d.data.generation.assets import trellis_voxelize
    output.mkdir(parents=True, exist_ok=True)
    mesh = trimesh.load(source, force='mesh', process=False)
    if not np.isfinite(mesh.vertices).all() or not len(mesh.faces):
        raise ValueError(f'Invalid source geometry: {source}')
    center = mesh.bounds.mean(axis=0)
    extent = float(np.ptp(mesh.vertices, axis=0).max())
    if extent <= 0:
        raise ValueError(f'Degenerate source geometry: {source}')
    mesh.apply_translation(-center)
    mesh.apply_scale(1.0 / extent)
    mesh.export(output / 'mesh.glb')
    mesh.export(output / 'voxel-source.ply')
    indices = trellis_voxelize(output / 'voxel-source.ply')
    if not len(indices):
        raise ValueError(f'Empty voxelization: {source}')
    grid = np.zeros((64, 64, 64), dtype=bool)
    grid[tuple(indices.T)] = True
    np.save(output / 'voxels.npy', grid)
    transform = {'source': str(source), 'translation_before_scale': (-center).tolist(),
                 'scale': 1.0 / extent, 'rotation': np.eye(3).tolist(),
                 'canonical_alignment_to_sam': 'requires_model_roundtrip_validation'}
    (output / 'normalization.json').write_text(json.dumps(transform, indent=2))
    return mesh, transform


def prepare_scan_mesh(task: tuple[str, str]) -> None:
    source, output = Path(task[0]), Path(task[1])
    marker = output / 'prepared.json'
    if marker.exists():
        return
    normalize_mesh(source, output)
    marker.write_text(json.dumps({'source': str(source), 'normalization': 'aabb_unit_cube'}))


def prepare_omniobject(root: str, output: str, seed: int = 42, validation_fraction: float = 0.1,
                      workers: int = 3) -> dict[str, Any]:
    base, destination = Path(root), Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    source_meshes: dict[str, Path] = {}
    for path in (base / 'raw_scans').rglob('*.obj'):
        if path.name.lower() in {'scan.obj', 'model.obj'}:
            identity = path.parent.parent.name if path.parent.name.lower() == 'scan' else path.parent.name
            if identity in source_meshes:
                raise ValueError(f'Ambiguous mesh for {identity}')
            source_meshes[identity] = path
    rows: list[ScanRecord] = []
    missing: list[str] = []
    captures = sorted((base / 'videos_processed').rglob('standard/images'))
    tasks = [(str(source_meshes[images.parent.parent.name]),
              str(destination / 'assets' / images.parent.parent.name))
             for images in captures if images.parent.parent.name in source_meshes]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for _ in pool.map(prepare_scan_mesh, tasks):
            pass
    for images in captures:
        identity = images.parent.parent.name
        if identity not in source_meshes:
            missing.append(identity)
            continue
        frames = sorted(path for path in images.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
        masks = [images.parent / 'matting' / path.name for path in frames]
        if not frames or not all(path.is_file() for path in masks):
            raise ValueError(f'Incomplete video/masks: {identity}')
        # Source archives and extracted files remain where they were downloaded. The
        # dataset root owns read-only symlinks, so loading does not duplicate a TB.
        video_link = destination / 'captures' / identity
        video_link.parent.mkdir(exist_ok=True)
        if not video_link.exists():
            video_link.symlink_to(images.parent.resolve(), target_is_directory=True)
        row = ScanRecord(dataset='omniobject3d', object_id=identity,
                         category=identity.rsplit('_', 1)[0],
                         split=object_split(identity, identity.rsplit('_', 1)[0], seed, validation_fraction),
                         frames=[f'captures/{identity}/images/{p.name}' for p in frames],
                         masks=[f'captures/{identity}/matting/{p.name}' for p in masks],
                         mesh=f'assets/{identity}/mesh.glb', voxels=f'assets/{identity}/voxels.npy')
        rows.append(row)
    if not rows:
        raise ValueError('No matched OmniObject3D video/mesh pairs')
    write_manifest(destination, rows)
    result = {'objects': len(rows), 'missing_scan_objects': missing,
              'frames': sum(len(row.frames) for row in rows), 'expected_catalogue_objects': 6000,
              'complete_6000': len(rows) >= 6000 and not missing}
    (destination / 'preparation.json').write_text(json.dumps(result, indent=2))
    return result


def write_manifest(root: Path, rows: list[ScanRecord]) -> None:
    path = root / 'manifest.jsonl'
    temporary = path.with_suffix('.jsonl.tmp')
    temporary.write_text(''.join(row.model_dump_json() + '\n' for row in sorted(rows, key=lambda row: row.object_id)))
    temporary.replace(path)


def render_gso_object(task: tuple[str, str, int, int]) -> dict[str, Any]:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
    os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
    os.environ['LP_NUM_THREADS'] = '2'
    import pyrender
    from video3d.data.generation.assets import asset_id_for, stage_gso
    source, output, frames, resolution = task
    model_dir = Path(source)
    identity = asset_id_for(model_dir.name)
    root = Path(output)
    asset = root / 'assets' / identity
    result = asset / 'record.json'
    if result.exists():
        record = json.loads(result.read_text())
        camera_data = json.loads((asset / 'cameras.json').read_text())
        if len(record['frames']) != frames or camera_data['resolution'] != resolution:
            raise ValueError(f'Render configuration differs for {identity}; use a different output root')
        return record
    staged = stage_gso(model_dir, asset / 'source')
    mesh, transform = normalize_mesh(staged, asset)
    scene = pyrender.Scene(bg_color=[0.93, 0.94, 0.95, 1.0], ambient_light=[0.35]*3)
    scene.add(pyrender.Mesh.from_trimesh(mesh, smooth=False))
    camera = scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4))
    light = scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.0))
    renderer = pyrender.OffscreenRenderer(resolution, resolution)
    images, masks, poses = [], [], []
    # AABB normalization does not bound the radius by 0.5. Fit the entire mesh's
    # bounding sphere inside the perspective frustum at every orbit angle.
    distance = max(2.05, float(np.linalg.norm(mesh.vertices, axis=1).max()) / np.sin(np.pi / 8) * 1.1)
    for index in range(frames):
        angle = 2 * np.pi * index / frames
        eye = np.array([1.9*np.cos(angle), 1.9*np.sin(angle), 0.75])
        eye *= distance / np.linalg.norm(eye)
        z_axis = eye / np.linalg.norm(eye)
        x_axis = np.cross(np.array([0., 0., 1.]), z_axis)
        x_axis /= np.linalg.norm(x_axis)
        pose = np.eye(4)
        pose[:3, :3] = np.column_stack([x_axis, np.cross(z_axis, x_axis), z_axis])
        pose[:3, 3] = eye
        scene.set_pose(camera, pose)
        scene.set_pose(light, pose)
        rgb, depth = renderer.render(scene)
        mask = depth > 0
        if not mask.any() or mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any():
            raise ValueError(f'Empty/clipped render: {identity}/{index}')
        image_path = asset / f'rgb-{index:03d}.jpg'
        mask_path = asset / f'mask-{index:03d}.png'
        Image.fromarray(rgb).save(image_path, quality=95)
        Image.fromarray(mask.astype(np.uint8)*255).save(mask_path)
        images.append(str(image_path.relative_to(root)))
        masks.append(str(mask_path.relative_to(root)))
        poses.append(pose.tolist())
    renderer.delete()
    focal = resolution / (2*np.tan(np.pi/8))
    camera_data = {'schema': 1, 'convention': 'opengl_camera_to_normalized_object',
                   'T_object_camera': poses, 'K': [[focal,0,resolution/2],[0,focal,resolution/2],[0,0,1]],
                   'resolution': resolution, 'source_normalization': transform,
                   'official_gso30_protocol': False}
    (asset / 'cameras.json').write_text(json.dumps(camera_data))
    record = ScanRecord(dataset='gso', object_id=identity, category='household', split='eval',
                        frames=images, masks=masks, mesh=f'assets/{identity}/mesh.glb',
                        voxels=f'assets/{identity}/voxels.npy', cameras=f'assets/{identity}/cameras.json')
    result.write_text(record.model_dump_json())
    print(json.dumps({'gso_render_complete': identity}), flush=True)
    return record.model_dump()


def prepare_gso(root: str, output: str, frames: int = 32, resolution: int = 384, workers: int = 3, limit: int = 0) -> dict[str, int]:
    raw, destination = Path(root), Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    snapshot_path = Path(__file__).with_name('gso-evaluation-objects.json')
    names = json.loads(snapshot_path.read_text())['objects']
    models = [raw / name for name in names if (raw / name / '.done').exists()]
    if limit:
        models = models[:limit]
    if not models:
        raise ValueError('No downloaded GSO models')
    tasks = [(str(path), str(destination), frames, resolution) for path in models]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        rows = [ScanRecord.model_validate(row) for row in pool.map(render_gso_object, tasks)]
    write_manifest(destination, rows)
    return {'objects': len(rows), 'frames': frames*len(rows)}
