"""Occlusion-aware surface exposure estimated from the exported voxel geometry."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import trimesh
from skimage.measure import marching_cubes

from video3d.data import reader
from video3d.data.targets import fill_voxels
from video3d.utils import sample_seed


def exposure_path(cfg: Any, clip_id: str, object_id: int) -> Path:
    from video3d.provenance import data_identity
    return Path(cfg.paths.cache_root) / "visibility" / data_identity(cfg) / f"{clip_id}-object-{object_id}.npz"


def voxel_mesh(voxels: np.ndarray) -> trimesh.Trimesh:
    vertices, faces, _, _ = marching_cubes(np.pad(fill_voxels(voxels).astype(np.float32), 1), level=0.5)
    vertices = (vertices - 1 + 0.5) / np.array(voxels.shape) - 0.5
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def visible_surface_points(mesh: trimesh.Trimesh, points: np.ndarray, transform: np.ndarray,
                           intrinsics: np.ndarray, instance: np.ndarray, object_id: int) -> np.ndarray:
    """Require in-frame projection, the object's instance label, and a first-hit surface ray."""
    camera = points @ transform[:3, :3].T + transform[:3, 3]
    front = camera[:, 2] > 0
    pixel = np.zeros((len(points), 2), dtype=np.int64)
    pixel[front] = np.rint(reader.project(camera[front], intrinsics)).astype(np.int64)
    height, width = instance.shape
    inside = front & (pixel[:, 0] >= 0) & (pixel[:, 0] < width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < height)
    candidates = np.flatnonzero(inside)
    candidates = candidates[instance[pixel[candidates, 1], pixel[candidates, 0]] == object_id]
    visible = np.zeros(len(points), dtype=bool)
    if not len(candidates):
        return visible
    origin = np.linalg.inv(transform)[:3, 3]
    directions = points[candidates] - origin
    locations, ray_indices, _ = mesh.ray.intersects_location(
        np.broadcast_to(origin, directions.shape), directions, multiple_hits=False)
    first_distance = np.linalg.norm(locations - origin, axis=1)
    target_distance = np.linalg.norm(directions[ray_indices], axis=1)
    visible[candidates[ray_indices]] = np.abs(first_distance - target_distance) <= 1e-4
    return visible


def object_exposure(clip: reader.Clip, object_id: int, voxels: np.ndarray,
                    destination: Path, num_points: int = 10000) -> dict[str, Any]:
    """Use deterministic area-uniform surface samples, retaining every per-frame bit."""
    if destination.exists():
        with np.load(destination) as saved:
            return exposure_summary(saved["frame_count"], saved["visible"][::2].sum(0))
    mesh = voxel_mesh(voxels)
    points, _ = trimesh.sample.sample_surface(mesh, num_points, seed=sample_seed(clip.clip_id, object_id))
    observations: list[np.ndarray] = []
    for frame in range(clip.num_frames):
        rotation, translation, scale = reader.camera_frame_pose(clip, object_id, frame)
        transform = np.eye(4)
        transform[:3, :3] = rotation @ np.diag(scale)
        transform[:3, 3] = translation
        observations.append(visible_surface_points(mesh, points, transform, reader.intrinsics(clip),
                                                    reader.load_instance_mask(clip, frame), object_id))
    visible = np.stack(observations)
    counts = visible.sum(0).astype(np.uint8)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, points=points.astype(np.float32), visible=visible,
                            frame_count=counts, seen=counts > 0,
                            train_frame_count_even=visible[::2].sum(0).astype(np.uint8),
                            train_frame_count_odd=visible[1::2].sum(0).astype(np.uint8),
                            geometry="64^3 exported voxel marching-cubes surface",
                            method="area-uniform points; first-hit self-occlusion plus visible instance mask")
    temporary.replace(destination)
    return exposure_summary(counts, visible[::2].sum(0))


def exposure_summary(counts: np.ndarray, training_counts: np.ndarray) -> dict[str, Any]:
    return {"surface_fraction_seen":float((counts > 0).mean()),
            "mean_frames_seen":float(counts.mean()),
            "surface_fraction_seen_in_16_even_frames":float((training_counts > 0).mean()),
            "frame_count_histogram":np.bincount(counts.astype(int), minlength=33).tolist(),
            "visibility_geometry":"voxel-based estimate, not original mesh/depth ground truth"}
