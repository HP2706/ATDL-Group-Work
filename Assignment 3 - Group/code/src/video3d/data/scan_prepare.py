"""Prepare existing scanned assets and captured frames without synthesizing training videos."""
from __future__ import annotations

from concurrent.futures import ProcessPoolExecutor, as_completed
import fcntl
import hashlib
import json
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image
import trimesh
from tqdm.auto import tqdm

from video3d.data.scanned import ScanRecord, object_split, read_mask, read_rgb


class UnusableCaptureError(ValueError):
    """A real capture has insufficient usable annotated frames for training."""


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
    output.mkdir(parents=True, exist_ok=True)
    with (output / 'prepare.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = output / 'prepared.json'
        if marker.exists():
            return
        normalize_mesh(source, output)
        temporary = marker.with_suffix('.json.tmp')
        temporary.write_text(json.dumps({'source': str(source), 'normalization': 'aabb_unit_cube'}))
        temporary.replace(marker)


def prepare_available_scans(root: str, output: str, shard: int = 0, shards: int = 8,
                            max_seconds: int = 10800, once: bool = False) -> dict[str, Any]:
    """Prepare disjoint objects as verified scan archives arrive. The normal full
    preparation path later reuses these same locked, complete assets.
    """
    if not 0 <= shard < shards or max_seconds < 1:
        raise ValueError('Invalid preparation shard')
    base, destination = Path(root), Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    catalogue = json.loads((base / 'catalogue.json').read_text())['files']
    video_categories = {Path(row['path']).name for row in catalogue if row['path'].startswith('/raw/videos_processed/')}
    scans = [row for row in catalogue if row['path'].startswith('/raw/raw_scans/')
             and Path(row['path']).name in video_categories]
    done: set[str] = set()
    checked = 0
    started = time.monotonic()
    while time.monotonic() - started < max_seconds:
        for row in scans:
            category = Path(row['path']).name.removesuffix('.tar.gz')
            marker = base / 'archives/raw_scans' / (category + '.tar.gz.extracted.json')
            if category in done or not marker.exists():
                continue
            folders = list((base / 'raw_scans').glob(category + '_*'))
            wrapper = base / 'raw_scans' / category
            if wrapper.is_dir():
                folders.append(wrapper)
            for folder in sorted(folders):
                for mesh in sorted(folder.rglob('*.obj')):
                    if mesh.parent.name.lower() != 'scan' and mesh.name.lower() not in {'scan.obj', 'model.obj'}:
                        continue
                    identity = mesh.parent.parent.name if mesh.parent.name.lower() == 'scan' else mesh.parent.name
                    owner = int.from_bytes(hashlib.sha256(identity.encode()).digest()[:8], 'big') % shards
                    if owner != shard:
                        continue
                    prepare_scan_mesh((str(mesh), str(destination / 'assets' / identity)))
                    checked += 1
                    print(json.dumps({'mesh_ready': identity, 'shard': shard}), flush=True)
            done.add(category)
        report = {'shard': shard, 'checked_objects': checked, 'scan_archives_seen': len(done),
                  'scan_archives_total': len(scans), 'elapsed_seconds': time.monotonic()-started,
                  'complete': len(done) == len(scans)}
        status = destination / f'live-preparation-{shard:02d}.json'
        temporary = status.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(report, indent=2))
        temporary.replace(status)
        if report['complete'] or once:
            return report
        time.sleep(15)
    return report


def capture_frames(images: Path, report_dir: Path) -> list[Path]:
    """Serialize validation of one capture across preparation jobs."""
    report_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / 'capture-validation.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return capture_frame_contents(images, report_dir)


def capture_frame_contents(images: Path, report_dir: Path) -> list[Path]:
    """Retain usable RGB/mask pairs and record exclusions without changing raw data.

    Extracted source archives are immutable. File metadata invalidates this cache
    when a capture is augmented or repaired.
    """
    from video3d.storage import atomic_json
    candidates = sorted(path for path in images.iterdir() if path.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    masks = [images.parent / 'matting' / path.name for path in candidates]
    if not candidates:
        raise UnusableCaptureError(f'No video frames: {images.parent.parent.name}')
    signature = [(str(path), path.stat().st_size, path.stat().st_mtime_ns)
                 if path.is_file() else (str(path), None, None) for path in [*candidates, *masks]]
    fingerprint = hashlib.sha256(json.dumps(signature).encode()).hexdigest()
    report = report_dir / 'capture-validation.json'
    if report.exists():
        cached = json.loads(report.read_text())
        if cached['source_signature'] == fingerprint:
            if len(cached['valid_frames']) < 16:
                raise UnusableCaptureError(f'Capture has fewer than 16 usable frames: {images.parent.parent.name}')
            return [images / name for name in cached['valid_frames']]
    valid: list[Path] = []
    excluded: list[dict[str, str]] = []
    for frame, mask_path in zip(candidates, masks, strict=True):
        if not mask_path.is_file():
            excluded.append({'frame': frame.name, 'reason': 'missing_mask'})
            continue
        try:
            mask = read_mask(mask_path, 0.5)
            image = read_rgb(frame)
        except OSError as error:
            excluded.append({'frame': frame.name, 'reason': f'undecodable_image: {error}'})
            continue
        if not mask.any():
            excluded.append({'frame': frame.name, 'reason': 'empty_mask'})
            continue
        if image.shape[:2] != mask.shape:
            excluded.append({'frame': frame.name, 'reason': 'rgb_mask_size_mismatch'})
            continue
        valid.append(frame)
    atomic_json(report, {'source_signature': fingerprint, 'valid_frames': [path.name for path in valid],
                         'excluded': excluded, 'raw_frames': len(candidates),
                         'usable_for_video_training': len(valid) >= 16})
    if len(valid) < 16:
        raise UnusableCaptureError(f'Capture has fewer than 16 usable frames: {images.parent.parent.name}')
    return valid


def prepare_capture(task: tuple[str, str, str]) -> tuple[str, list[str]]:
    source, output, images = map(Path, task)
    prepare_scan_mesh((str(source), str(output)))
    frames = capture_frames(images, output)
    return images.parent.parent.name, [str(path) for path in frames]


def source_scan_meshes(root: Path) -> dict[str, Path]:
    """Scan directories also contain legitimate nonstandard names, such as wallet.obj."""
    meshes: dict[str, Path] = {}
    for path in root.rglob('*.obj'):
        if path.parent.name.lower() != 'scan' and path.name.lower() not in {'scan.obj', 'model.obj'}:
            continue
        identity = path.parent.parent.name if path.parent.name.lower() == 'scan' else path.parent.name
        if identity in meshes:
            raise ValueError(f'Ambiguous mesh for {identity}')
        meshes[identity] = path
    return meshes


def prepare_omniobject(root: str, output: str, seed: int = 42, validation_fraction: float = 0.1,
                      workers: int = 3, shard: int = -1, shards: int = 1) -> dict[str, Any]:
    from video3d.storage import atomic_json
    if workers < 1 or shards < 1 or (shard != -1 and not 0 <= shard < shards):
        raise ValueError('Invalid preparation worker/shard configuration')
    started = time.monotonic()
    base, destination = Path(root), Path(output)
    destination.mkdir(parents=True, exist_ok=True)
    source_meshes = source_scan_meshes(base / 'raw_scans')
    rows: list[ScanRecord] = []
    missing: list[str] = []
    multiobject: list[str] = []
    captures = sorted((base / 'videos_processed').rglob('standard/images'))
    tasks = [(str(source_meshes[images.parent.parent.name]),
              str(destination / 'assets' / images.parent.parent.name), str(images))
             for images in captures if images.parent.parent.name in source_meshes]
    if shard >= 0:
        tasks = [task for task in tasks if int.from_bytes(
            hashlib.sha256(Path(task[1]).name.encode()).digest()[:8], 'big') % shards == shard]
    status_path = destination / ('preparation-progress.json' if shard < 0 else f'preparation-shard-{shard:02d}.json')
    validated: dict[str, list[str]] = {}
    excluded: dict[str, str] = {}
    report: dict[str, Any] = {'shard': shard, 'shards': shards, 'workers': workers,
                              'objects_total': len(tasks), 'objects_complete': 0, 'state': 'RUNNING'}
    atomic_json(status_path, report)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(prepare_capture, task): Path(task[1]).name for task in tasks}
        for future in tqdm(as_completed(futures), total=len(tasks), desc=f'Prepare Omni shard {shard}', unit='object'):
            error = future.exception()
            if isinstance(error, UnusableCaptureError):
                excluded[futures[future]] = str(error)
            elif error is not None:
                atomic_json(status_path, {**report, 'state': 'FAILED', 'failed_object': futures[future],
                                         'error_type': type(error).__name__, 'error': str(error)})
                for pending in futures:
                    pending.cancel()
                raise RuntimeError(f'Preparation failed for {futures[future]}') from error
            else:
                identity, paths = future.result()
                validated[identity] = paths
            report.update(objects_complete=len(validated), objects_processed=len(validated)+len(excluded),
                          excluded_unusable_captures=excluded, elapsed_seconds=time.monotonic()-started)
            atomic_json(status_path, report)
    if shard >= 0:
        report.update(state='COMPLETE', frames=sum(len(paths) for paths in validated.values()))
        atomic_json(status_path, report)
        return report
    for images in captures:
        identity = images.parent.parent.name
        if identity in excluded:
            continue
        if identity not in source_meshes:
            (multiobject if ',' in identity else missing).append(identity)
            continue
        frames = [Path(path) for path in validated[identity]]
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
              'excluded_multiobject_captures': multiobject,
              'excluded_unusable_captures': excluded,
              'available_scan_objects': len(source_meshes),
              'all_retained_objects_have_mesh': True,
              'dataset_scope': 'available_matched_single_object_processed_captures',
              'frames': sum(len(row.frames) for row in rows), 'expected_catalogue_objects': 6000,
              'complete_6000': len(rows) >= 6000 and not missing}
    (destination / 'preparation.json').write_text(json.dumps(result, indent=2))
    atomic_json(status_path, {**report, 'state': 'COMPLETE', 'objects': len(rows)})
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
