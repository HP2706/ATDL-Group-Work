"""Surface diagnostics in explicit mesh coordinates, using the shared ICP implementation."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import trimesh

from video3d.metrics import icp_fit


def normalization(vertices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = vertices.min(axis=0), vertices.max(axis=0)
    center = (lo + hi) / 2
    scale = 2.0 / max(float((hi - lo).max()), 1e-8)
    matrix = np.eye(4)
    matrix[:3, :3] *= scale
    matrix[:3, 3] = -center * scale
    return (vertices - center) * scale, matrix


def transform_points(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return points @ matrix[:3, :3].T + matrix[:3, 3]


def ray_scene(mesh: trimesh.Trimesh) -> Any:
    import open3d as o3d
    scene = o3d.t.geometry.RaycastingScene(nthreads=4)
    scene.add_triangles(o3d.core.Tensor(np.asarray(mesh.vertices, dtype=np.float32)),
                        o3d.core.Tensor(np.asarray(mesh.faces, dtype=np.uint32)))
    return scene


def closest_surface(mesh: trimesh.Trimesh, points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Native BVH queries avoid trimesh's unbounded Python candidate arrays on dense scans."""
    import open3d as o3d
    result = ray_scene(mesh).compute_closest_points(o3d.core.Tensor(points.astype(np.float32)), nthreads=4)
    closest = result["points"].numpy().astype(np.float64)
    return closest, np.linalg.norm(points-closest, axis=1), result["primitive_ids"].numpy().astype(np.int64)


def visibility(mesh: trimesh.Trimesh, points: np.ndarray, camera: dict[str, Any],
               frame_ids: list[int]) -> np.ndarray:
    """Frustum and first-hit mesh occlusion, for registered OpenGL render cameras."""
    import open3d as o3d
    scene = ray_scene(mesh)
    visible = []
    intrinsic = np.asarray(camera['K'])
    resolution = camera['resolution']
    for frame in frame_ids:
        pose = np.asarray(camera['T_object_camera'][frame])
        local = transform_points(points, np.linalg.inv(pose))
        depth = -local[:, 2]
        # OpenGL camera +Y is up; image +Y is down.
        u = intrinsic[0, 0] * local[:, 0] / np.maximum(depth, 1e-8) + intrinsic[0, 2]
        v = -intrinsic[1, 1] * local[:, 1] / np.maximum(depth, 1e-8) + intrinsic[1, 2]
        frustum = (depth > 0) & (u >= 0) & (u < resolution) & (v >= 0) & (v < resolution)
        origin = pose[:3, 3]
        offsets = points - origin
        length = np.linalg.norm(offsets, axis=1)
        rays = np.column_stack([np.broadcast_to(origin, points.shape), offsets / length[:, None]])
        first = scene.cast_rays(o3d.core.Tensor(rays.astype(np.float32)), nthreads=4)["t_hit"].numpy()
        visible.append(frustum & (np.abs(first - length) < 0.002))
    return np.asarray(visible)


def cache_visibility(mesh_path: str, camera_path: str, frame_ids: list[int], output: str,
                     samples: int = 8192) -> dict[str, Any]:
    """Prediction-independent surface partition, cached once for a frozen evaluation clip."""
    import hashlib
    path = Path(output)
    signature = {"mesh_sha256": hashlib.sha256(Path(mesh_path).read_bytes()).hexdigest(),
        "camera_sha256": hashlib.sha256(Path(camera_path).read_bytes()).hexdigest(),
        "frame_ids": frame_ids, "samples": samples, "seed": 43, "ray_tolerance": .002}
    metadata = path.with_suffix(".json")
    if path.is_file() and metadata.is_file():
        if json.loads(metadata.read_text())["signature"] != signature:
            raise ValueError("Visibility cache protocol changed")
        return json.loads(metadata.read_text())
    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    points, faces = trimesh.sample.sample_surface(mesh, samples, seed=43)
    camera = json.loads(Path(camera_path).read_text())
    observed = visibility(mesh, points, camera, frame_ids)
    arrays = dict(points=points, faces=faces, observed_by_frame=observed,
        reference_visible=observed[0], newly_revealed=observed[1:].any(axis=0) & ~observed[0],
        never_observed=~observed.any(axis=0))
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)
    result = {"signature": signature, "visible_surface_fraction_per_frame": observed.mean(axis=1).tolist(),
        "reference_visible_fraction": float(arrays["reference_visible"].mean()),
        "newly_revealed_fraction": float(arrays["newly_revealed"].mean()),
        "never_observed_fraction": float(arrays["never_observed"].mean()),
        "occlusion_kind": "self-occlusion; isolated GSO render, no external occluders"}
    metadata.write_text(json.dumps(result, indent=2)+"\n")
    return result


def analyze(directory: str, samples: int = 8192) -> dict[str, Any]:
    root = Path(directory)
    arrays = np.load(root / 'mesh-exact.npz')
    pred = trimesh.Trimesh(vertices=arrays['vertices'], faces=arrays['faces'], process=False)
    gt = trimesh.load(root / 'ground-truth.glb', force='mesh', process=False)
    if not len(gt.faces):
        raise ValueError('Ground truth mesh is empty')
    protocol = json.loads((root / 'protocol.json').read_text())
    pred_points, pred_faces = trimesh.sample.sample_surface(pred, samples, seed=42)
    cache = np.load(protocol['visibility_cache']) if protocol.get('visibility_cache') else None
    if cache is not None:
        gt_points, gt_faces = cache['points'], cache['faces']
        if len(gt_points) != samples:
            raise ValueError("Surface evaluation and visibility cache sample counts differ")
    else:
        gt_points, gt_faces = trimesh.sample.sample_surface(gt, samples, seed=43)
    _, pred_norm = normalization(pred.vertices)
    _, gt_norm = normalization(gt.vertices)
    align = icp_fit(torch.tensor(transform_points(pred_points, pred_norm), device='cuda', dtype=torch.float32),
                    torch.tensor(transform_points(gt_points, gt_norm), device='cuda', dtype=torch.float32),
                    iterations=100, estimate_scale=True)
    basis = align(torch.cat([torch.zeros(1, 3, device='cuda'), torch.eye(3, device='cuda')])).cpu().numpy()
    fit = np.eye(4)
    fit[:3, :3] = (basis[1:] - basis[0]).T
    fit[:3, 3] = basis[0]
    matrix = np.linalg.inv(gt_norm) @ fit @ pred_norm
    edge_counts = np.bincount(pred.edges_unique_inverse)
    diagnostics: dict[str, Any] = dict(samples_per_direction=samples, distance_units='GT normalized longest side = 1',
        alignment='bbox normalization + single-start similarity ICP, 100 iterations',
        raw_prediction_frame='SAM 3D decoder coordinates; physical scale not asserted',
        prediction_vertices=len(pred.vertices), prediction_faces=len(pred.faces),
        prediction_watertight=bool(pred.is_watertight), prediction_euler_number=int(pred.euler_number),
        boundary_edges=int((edge_counts == 1).sum()), nonmanifold_edges=int((edge_counts > 2).sum()),
        degenerate_faces=int((pred.area_faces < 1e-12).sum()),
        winding_consistent=bool(pred.is_winding_consistent), surface_area=float(pred.area),
        ground_truth_watertight=bool(gt.is_watertight),
        vertex_bounds=pred.bounds.tolist(), ground_truth_bounds=gt.bounds.tolist(),
        prediction_to_gt_column_vector_4x4=matrix.tolist(),
        caveat='Distances are exact closest-triangle distances at sampled points. ICP is not guaranteed globally optimal; scan incompleteness is not model error.')
    fields: dict[str, np.ndarray] = dict(prediction_points=pred_points, prediction_faces=pred_faces,
        ground_truth_points=gt_points, ground_truth_faces=gt_faces, prediction_to_gt=matrix)
    for label, transform in [('canonical', np.eye(4)), ('aligned', matrix)]:
        vertices = transform_points(pred.vertices, transform)
        mesh = trimesh.Trimesh(vertices=vertices, faces=pred.faces, process=False)
        points = transform_points(pred_points, transform)
        closest_gt, distance_pred, triangle_gt = closest_surface(gt, points)
        closest_pred, distance_gt, triangle_pred = closest_surface(mesh, gt_points)
        fields.update({label+'_prediction_distance': distance_pred, label+'_ground_truth_distance': distance_gt,
            label+'_closest_gt_points': closest_gt, label+'_closest_pred_points': closest_pred,
            label+'_closest_gt_faces': triangle_gt, label+'_closest_pred_faces': triangle_pred})
        normals = np.abs((mesh.face_normals[pred_faces] * gt.face_normals[triangle_gt]).sum(axis=1))
        metrics: dict[str, Any] = dict(prediction_to_gt_mean=float(distance_pred.mean()),
            gt_to_prediction_mean=float(distance_gt.mean()), chamfer_l1_sum=float(distance_pred.mean()+distance_gt.mean()),
            prediction_p95=float(np.quantile(distance_pred, .95)), gt_p95=float(np.quantile(distance_gt, .95)),
            absolute_normal_consistency=float(normals.mean()))
        for threshold in [.01, .02, .05]:
            precision, recall = float((distance_pred < threshold).mean()), float((distance_gt < threshold).mean())
            metrics[str(threshold)] = dict(precision=precision, recall=recall,
                fscore=2*precision*recall/(precision+recall) if precision+recall else 0.0)
        diagnostics[label] = metrics
        for name, cloud, distances in [('spurious', points, distance_pred), ('missing', gt_points, distance_gt)]:
            intensity = np.minimum(distances / .05, 1)
            colors = np.stack([255*intensity, 180*(1-intensity), 80*(1-intensity), np.full(samples, 255)], axis=1).astype(np.uint8)
            trimesh.points.PointCloud(cloud, colors=colors).export(root / f'{label}-{name}-error.ply')
    protocol = json.loads((root / 'protocol.json').read_text())
    if protocol['camera_registration'] == 'known_render_poses':
        if cache is not None:
            regions = {name: cache[name] for name in ['reference_visible', 'newly_revealed', 'never_observed']}
        else:
            camera = json.loads((root / 'cameras.json').read_text())
            observed = visibility(gt, gt_points, camera, protocol['frame_ids'])
            regions = dict(reference_visible=observed[0], newly_revealed=observed[1:].any(axis=0) & ~observed[0],
                           never_observed=~observed.any(axis=0))
        diagnostics['visibility'] = {'partition': 'common evaluation clip, independent of method input budget'}
        for name, mask in regions.items():
            fields[name] = mask
            distances = fields['aligned_ground_truth_distance'][mask]
            diagnostics['visibility'][name] = dict(count=int(mask.sum()),
                completeness_distance=float(distances.mean()) if len(distances) else None,
                recall_002=float((distances < .02).mean()) if len(distances) else None)
    else:
        diagnostics['visibility'] = 'unavailable: mesh-camera registration unverified'
    np.savez_compressed(root / 'surface-errors.npz', **fields)
    (root / 'surface-metrics.json').write_text(json.dumps(diagnostics, indent=2)+'\n')
    return diagnostics
