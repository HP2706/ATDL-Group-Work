"""Exact mask occlusion and explicitly approximate voxel-observation metrics.

The training export contains no meshes, scene depth or amodal images. Its stored
visible/amodal pixel counts give exact rendered occlusion. The optional geometry
backend traces rays from the *same* surface-voxel centres in every frame.
These are fractions of occupied voxels, not surface
area estimates or ground-truth mesh visibility. Nothing here is a model input.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from video3d.data import reader


def coverage_summary(observed: NDArray[np.bool_], alone: NDArray[np.bool_],
                     reference: int) -> dict[str, Any]:
    """Aggregate persistent element identities, with undefined recovery as None."""
    if observed.ndim != 2 or observed.shape != alone.shape or not observed.shape[1]:
        raise ValueError("Expected matching nonempty (frames, elements) visibility arrays")
    if not 0 <= reference < len(observed) or np.any(observed & ~alone):
        raise ValueError("Invalid reference or scene visibility outside target-alone visibility")
    union = observed.any(axis=0)
    coverage = observed.mean(axis=1)
    hidden = alone & ~observed
    hidden_counts = hidden.sum(axis=1)
    recovered_counts = (hidden & union[None]).sum(axis=1)
    recovery = [float(n / d) if d else None for n, d in zip(recovered_counts, hidden_counts)]
    gains = (union[None] & ~observed).mean(axis=1)
    return {
        "reference_frame": reference,
        "elements": observed.shape[1],
        "total_observed_fraction": float(union.mean()),
        "reference_observed_fraction": float(coverage[reference]),
        "additional_fraction": float(gains[reference]),
        "best_frame_gain_fraction": float(union.mean() - coverage.max()),
        "external_occlusion_recovery_fraction": recovery[reference],
        "reference_hidden_elements": int(hidden_counts[reference]),
        "per_frame_observed_fraction": coverage.tolist(),
        "per_reference_additional_fraction": gains.tolist(),
        "per_reference_external_occlusion_recovery_fraction": recovery,
    }


def project_voxels(points: NDArray[np.float64], rotation: NDArray[np.float64],
                   translation: NDArray[np.float64], scale: NDArray[np.float64],
                   intrinsics: NDArray[np.float64], image_size: tuple[int, int],
                   resolution: int) -> tuple[NDArray[np.int64], NDArray[np.float64], NDArray[np.bool_]]:
    """Project canonical centres to a diagnostic raster; reject behind/outside points."""
    camera = (points * scale) @ rotation.T + translation
    height, width = image_size
    uv = np.zeros((len(points), 2), dtype=np.float64)
    front = camera[:, 2] > 0
    uv[front] = reader.project(camera[front], intrinsics)
    raster_height = max(1, round(height * resolution / width))
    pixel = np.floor((uv + 0.5) * resolution / width).astype(np.int64)
    inside = front & (pixel[:, 0] >= 0) & (pixel[:, 0] < resolution)
    inside &= (pixel[:, 1] >= 0) & (pixel[:, 1] < raster_height)
    flat = np.clip(pixel[:, 1], 0, raster_height - 1) * resolution + np.clip(pixel[:, 0], 0, resolution - 1)
    return flat, camera[:, 2], inside


def voxel_self_visibility(voxels: NDArray[np.bool_], camera: NDArray[np.float64],
                          points: NDArray[np.float64]) -> NDArray[np.bool_]:
    """Exact grid traversal from each occupied centre towards the canonical camera.

    The starting cell is skipped. Any other occupied cell blocks the ray;
    leaving the grid establishes visibility. This has no diagnostic raster or
    depth-tolerance parameter. It is still an approximation to the source mesh.
    """
    size = voxels.shape[0]
    cells = np.floor((points + 0.5) * size).astype(np.int64)
    direction = camera[None] - points
    steps = np.sign(direction).astype(np.int64)
    delta = np.divide(1 / size, np.abs(direction), out=np.full_like(direction, np.inf), where=direction != 0)
    crossing = delta * 0.5
    active = np.arange(len(points))
    visible = np.zeros(len(points), bool)
    for _ in range(3 * size + 1):
        if not len(active):
            return visible
        next_t = crossing[active].min(axis=1)
        camera_reached = next_t >= 1
        visible[active[camera_reached]] = True
        active = active[~camera_reached]
        next_t = next_t[~camera_reached]
        axes = np.isclose(crossing[active], next_t[:, None], atol=1e-12, rtol=0)
        cells[active] += steps[active] * axes
        # Avoid inf * 0 for rays parallel to a grid axis.
        rows, columns = np.nonzero(axes)
        crossing[active[rows], columns] += delta[active[rows], columns]
        inside = ((cells[active] >= 0) & (cells[active] < size)).all(axis=1)
        visible[active[~inside]] = True
        active = active[inside]
        blocked = voxels[tuple(cells[active].T)]
        active = active[~blocked]
    raise RuntimeError("Voxel traversal failed to leave a finite grid")


def measure_clip(clip: reader.Clip, voxels_file: Path, reference: int | None = None) -> dict[str, Any]:
    """Measure every frame and object; use the middle frame unless specified."""
    reference = clip.num_frames // 2 if reference is None else reference
    if not 0 <= reference < clip.num_frames:
        raise ValueError("Reference frame outside clip")
    visible_px = reader.visibility(clip, "visible_px").astype(np.int64)
    amodal_px = reader.visibility(clip, "amodal_px").astype(np.int64)
    if np.any(visible_px < 0) or np.any(amodal_px <= 0) or np.any(visible_px > amodal_px):
        raise ValueError(f"{clip.clip_id}: invalid visible/amodal pixel counts")
    visible_fraction = visible_px / amodal_px
    if not np.allclose(visible_fraction, reader.visibility(clip), atol=1e-7):
        raise ValueError(f"{clip.clip_id}: visibility metadata disagrees with pixel counts")
    entries = [reader.object_entry(clip, k) for k in range(1, clip.num_objects + 1)]
    grids = [reader.load_voxels(voxels_file, entry["asset_id"]) for entry in entries]
    centres = [reader.voxel_centers(grid) for grid in grids]
    if any(len(points) == 0 for points in centres):
        raise ValueError("Cannot measure an empty voxel grid")
    observed: list[list[NDArray[np.bool_]]] = [[] for _ in entries]
    alone: list[list[NDArray[np.bool_]]] = [[] for _ in entries]
    height, width = clip.image_size
    for t in range(clip.num_frames):
        labels = reader.load_instance_mask(clip, t)
        counts = np.bincount(labels.ravel(), minlength=clip.num_objects + 1)
        if labels.shape != (height, width) or labels.max() > clip.num_objects or not np.array_equal(counts[1:], visible_px[t]):
            raise ValueError(f"{clip.clip_id}/{t}: dense instance masks disagree with metadata")
        for k, points in enumerate(centres, 1):
            rotation, translation, scale = reader.camera_frame_pose(clip, k, t)
            pixel, _, inside = project_voxels(points, rotation, translation, scale, reader.intrinsics(clip), clip.image_size, width)
            camera = (-translation @ rotation) / scale
            solo = inside & voxel_self_visibility(grids[k - 1], camera, points)
            # Source masks determine external occlusion; background at a coarse
            # voxel boundary is not an occluder. Exact zero occlusion overrides
            # coarse silhouette discrepancies entirely.
            pixel_labels = labels.ravel()[pixel]
            external = (pixel_labels > 0) & (pixel_labels != k)
            if visible_px[t, k - 1] == amodal_px[t, k - 1]:
                external[:] = False
            observed[k - 1].append(solo & ~external)
            alone[k - 1].append(solo)
    objects: list[dict[str, Any]] = []
    for i, entry in enumerate(entries):
        occlusion = 1 - visible_fraction[:, i]
        objects.append({
            "object_id": entry["k"], "asset_id": entry["asset_id"],
            "reference_frame": reference,
            "mean_occlusion": float(occlusion.mean()), "min_occlusion": float(occlusion.min()),
            "max_occlusion": float(occlusion.max()), "reference_occlusion": float(occlusion[reference]),
            "frames_at_least_10pct_occluded": int((occlusion >= 0.1).sum()),
            "min_visible_pixels": int(visible_px[:, i].min()),
            "per_frame": [{"frame": t, "visible_px": int(visible_px[t, i]),
                           "amodal_px": int(amodal_px[t, i]), "visible_fraction": float(visible_fraction[t, i]),
                           "occlusion_fraction": float(occlusion[t])} for t in range(clip.num_frames)],
            "voxel_proxy": coverage_summary(np.stack(observed[i]), np.stack(alone[i]), reference),
            "most_occluded_frame_voxel_proxy": coverage_summary(np.stack(observed[i]), np.stack(alone[i]), int(occlusion.argmax())),
        })
    geometry_keys = ["total_observed_fraction", "additional_fraction", "best_frame_gain_fraction"]
    summary = {"mean_occlusion": float((1 - visible_fraction).mean()),
               "max_occlusion": float((1 - visible_fraction).max()),
               "frames_with_any_object_at_least_10pct_occluded": int(((1 - visible_fraction) >= 0.1).any(axis=1).sum()),
               "reference_occlusion": float((1 - visible_fraction[reference]).mean())}
    summary.update({f"voxel_proxy_{key}": float(np.mean([obj["voxel_proxy"][key] for obj in objects])) for key in geometry_keys})
    recovery = [obj["voxel_proxy"]["external_occlusion_recovery_fraction"] for obj in objects
                if obj["voxel_proxy"]["external_occlusion_recovery_fraction"] is not None]
    summary["voxel_proxy_external_occlusion_recovery_fraction"] = float(np.mean(recovery)) if recovery else None
    summary["objects_with_proxy_external_occlusion"] = len(recovery)
    return {"clip_id": clip.clip_id, "num_frames": clip.num_frames, "num_objects": clip.num_objects,
            "reference_frame": reference, "arc_span_deg": reader.arc_span_deg(clip),
            "summary": summary, "objects": objects}


def video_metrics(split_dir: str, out: str, reference: int | None = None,
                  cache_dir: str | None = None) -> dict[str, Any]:
    """Fire entrypoint: write JSON, equal-object-weighted clip CSV, and object CSV."""
    data = reader.open_split(split_dir, cache_dir)
    clips = [measure_clip(data.load_clip(clip_id), data.voxels_file, reference) for clip_id in data.clips]
    report = {
        "schema_version": 1, "split_dir": str(Path(split_dir).resolve()),
        "source": data.index.get("preview_source"),
        "mask_kind": "dense rendered per-pixel instance IDs; 0=background, k=object k; no point prompts",
        "occlusion_method": "1 - visible_px / amodal_px; exact rendered in-image metadata, masks verified on every frame",
        "geometry_method": "surface_voxel_centre_ray_traversal_proxy",
        "external_occlusion_method": "native-resolution rendered instance masks at projected voxel centres",
        "limitations": ["Voxel fractions are not mesh surface area; 64^3 quantization affects thin structures and silhouette boundaries.",
                        "No meshes, scene depth or amodal images are included in this training export.",
                        "Amodal pixel counts are clipped to the image; truncation is not external occlusion.",
                        "Clip summaries weight objects equally; recovery excludes objects with no hidden reference samples.",
                        "Reference defaults to middle frame; all-reference gains and recovery are saved per object."],
        "clips": clips,
    }
    output = Path(out)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    clip_rows = [{"clip_id": clip["clip_id"], "num_objects": clip["num_objects"],
                  "arc_span_deg": clip["arc_span_deg"], **clip["summary"]} for clip in clips]
    object_rows = [{"clip_id": clip["clip_id"], **{key: value for key, value in obj.items() if key not in {"per_frame", "voxel_proxy", "most_occluded_frame_voxel_proxy"}},
                    **{f"voxel_proxy_{key}": value for key, value in obj["voxel_proxy"].items() if not key.startswith("per_")}}
                   for clip in clips for obj in clip["objects"]]
    for suffix, rows in (("clips", clip_rows), ("objects", object_rows)):
        with output.with_suffix(f".{suffix}.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
            writer.writeheader()
            writer.writerows(rows)
    return {"clips": len(clips), "objects": len(object_rows), "report": str(output.resolve())}
