"""Controlled target/occluder placement and mesh-based evidence checks."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .config import Config
from .conventions import intrinsics as make_intrinsics, object_matrix, project, transform_points
from .precheck import estimate_visibility
from .sampling import ObjectPose, Trajectory, sample_layout, sample_trajectory, scene_extent


def sample_occlusion_scene(rng: np.random.Generator, cfg: Config, asset_ids: list[str],
                           centres: list[NDArray[np.float64]]) -> tuple[list[ObjectPose], Trajectory]:
    """Separate objects in depth while overlapping their reference projections."""
    if len(asset_ids) != 2:
        raise ValueError("recoverable occlusion currently requires exactly two objects")
    objects = sample_layout(rng, cfg, asset_ids)
    objects[1].s_wo = objects[0].s_wo * cfg.occluder_scale_ratio
    azimuth = float(rng.uniform(0, 360))
    elevation = float(rng.uniform(*cfg.start_elevation_range_deg))
    az, el = np.radians([azimuth, elevation])
    direction = np.array([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)])
    right = np.array([-np.sin(az), np.cos(az), 0.])
    distance = objects[0].bounding_radius + objects[1].bounding_radius + 0.08
    lateral = 0.4 * objects[0].s_wo[0]
    sign = 1 if rng.random() < 0.5 else -1
    separation = direction * distance + right * lateral
    objects[0].t_wo = -0.5 * separation
    objects[1].t_wo = 0.5 * separation
    center, radius = scene_extent(objects, centres)
    trajectory = sample_trajectory(rng, cfg, center, radius)
    trajectory.start_azimuth_deg = azimuth - 0.5 * trajectory.arc_span_deg
    trajectory.start_elevation_deg = elevation - 0.5 * trajectory.elevation_drift_deg
    factor = trajectory.orbit_radius / radius
    low, high = 0., 1.8 * float(objects[0].s_wo[0])
    desired = float(rng.uniform(0.40, 0.60))
    intrinsic = make_intrinsics(cfg.width, cfg.height, cfg.hfov_deg)
    # First fit the *projected* occluder size. A fixed physical scale ratio
    # cannot handle different silhouettes (e.g. a narrow bottle vs a broad box).
    ratio_low, ratio_high = cfg.occluder_scale_ratio_range
    for _ in range(8):
        ratio = 0.5 * (ratio_low + ratio_high)
        objects[1].s_wo = objects[0].s_wo * ratio
        distance = objects[0].bounding_radius + objects[1].bounding_radius + 0.08
        separation = direction * distance
        objects[0].t_wo, objects[1].t_wo = -0.5 * separation, 0.5 * separation
        center, radius = scene_extent(objects, centres)
        trajectory.center, trajectory.orbit_radius = center, radius * factor
        reference = trajectory.camera_at(trajectory.eased()[cfg.num_frames // 2])
        visible, amodal = estimate_visibility(intrinsic, reference, objects, centres,
                                               cfg.width, cfg.height, cfg.precheck_resolution)
        maximum_occlusion = 1 - visible[0] / amodal[0] if amodal[0] else 1.
        if abs(maximum_occlusion - 0.65) < 0.04:
            break
        if maximum_occlusion < 0.65:
            ratio_low = ratio
        else:
            ratio_high = ratio
    for _ in range(8):
        lateral = 0.5 * (low + high)
        separation = direction * distance + right * lateral * sign
        objects[0].t_wo = -0.5 * separation
        objects[1].t_wo = 0.5 * separation
        center, radius = scene_extent(objects, centres)
        trajectory.center = center
        trajectory.orbit_radius = radius * factor
        reference = trajectory.camera_at(trajectory.eased()[cfg.num_frames // 2])
        visible, amodal = estimate_visibility(intrinsic, reference, objects, centres,
                                               cfg.width, cfg.height, cfg.precheck_resolution)
        fraction = visible[0] / amodal[0] if amodal[0] else 0.
        if abs(fraction - desired) < 0.025:
            break
        if fraction < desired:
            low = lateral
        else:
            high = lateral
    return objects, trajectory


def occlusion_failures(visible_fraction: NDArray[np.float64], cfg: Config) -> list[str]:
    """Apply temporal conditions to target k=1; other objects remain valid inputs."""
    if cfg.occlusion_mode == "random":
        return []
    if cfg.occlusion_mode != "recoverable":
        raise ValueError(f"Unknown occlusion mode {cfg.occlusion_mode}")
    target = visible_fraction[:, 0]
    reference = len(target) // 2
    occlusion = 1 - target
    low, high = cfg.occlusion_reference_range
    failures: list[str] = []
    if not low <= occlusion[reference] <= high:
        failures.append(f"reference occlusion {occlusion[reference]:.3f} outside [{low}, {high}]")
    count = int((occlusion >= cfg.occlusion_frame_threshold).sum())
    if count < cfg.occlusion_min_frames:
        failures.append(f"only {count} frames have sufficient occlusion")
    if target.max() < cfg.occlusion_min_best_visibility:
        failures.append(f"best visibility {target.max():.3f} below {cfg.occlusion_min_best_visibility}")
    return failures


def mesh_evidence_metrics(cfg: Config, objects: list[ObjectPose], frames: list[NDArray[np.float64]],
                          intrinsics: NDArray[np.float64], rendered: dict[str, NDArray[Any]],
                          seed: int, split: str = "train") -> dict[str, Any]:
    """Area-uniform mesh samples, scene depth and target-alone raycasting.

    Counts estimate surface area, not exact integrals. The same identities are
    used across all frames. Amodal/depth/mesh data remain annotation-only.
    """
    from .validate import load_canonical_mesh, raycast_object, sample_surface
    target = objects[0]
    mesh = load_canonical_mesh(Path(cfg.dataset_root) / "assets" / split / target.asset_id / "mesh.glb")
    points, _ = sample_surface(mesh, cfg.occlusion_surface_samples, seed)
    pose = {"R_wo": target.R_wo, "t_wo": target.t_wo, "s_wo": target.s_wo}
    reference = len(frames) // 2
    _, solo_depth = raycast_object(mesh, pose, intrinsics, frames[reference], cfg.width, cfg.height)
    observed: list[NDArray[np.bool_]] = []
    reference_alone = np.zeros(len(points), bool)
    tolerance = 0.01 * float(target.s_wo.max())
    for t, extrinsics in enumerate(frames):
        camera = transform_points(extrinsics @ object_matrix(target.R_wo, target.t_wo, target.s_wo), points)
        uv, z = project(intrinsics, camera)
        pixel = np.floor(uv + 0.5).astype(int)
        inside = (z > 0) & (pixel[:, 0] >= 0) & (pixel[:, 0] < cfg.width) & (pixel[:, 1] >= 0) & (pixel[:, 1] < cfg.height)
        x, y = np.clip(pixel[:, 0], 0, cfg.width - 1), np.clip(pixel[:, 1], 0, cfg.height - 1)
        seen = inside & (rendered["visible"][t, y, x] == 1)
        seen &= np.abs(rendered["depth"][t, y, x] - z) <= tolerance
        observed.append(seen)
        if t == reference:
            reference_alone = inside & (np.abs(solo_depth[y, x] - z) <= tolerance)
    visibility = np.stack(observed)
    union = visibility.any(axis=0)
    hidden = reference_alone & ~visibility[reference]
    recovery = float((hidden & union).sum() / hidden.sum()) if hidden.any() else None
    return {"target_object_id": 1, "reference_frame": reference,
            "method": "area-uniform source-mesh samples with depth tests", "surface_samples": len(points),
            "surface_coverage": float(union.mean()), "reference_surface_coverage": float(visibility[reference].mean()),
            "additional_surface_coverage": float((union & ~visibility[reference]).mean()),
            "best_frame_gain": float(union.mean() - visibility.mean(axis=1).max()),
            "reference_hidden_samples": int(hidden.sum()), "external_occlusion_recovery": recovery,
            "per_frame_surface_coverage": visibility.mean(axis=1).tolist(), "depth_tolerance": tolerance}
