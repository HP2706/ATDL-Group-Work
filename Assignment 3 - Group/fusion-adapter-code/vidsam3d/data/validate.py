"""`validate-data`: check that an export folder holds exactly what the pipeline reads.

Per clip: all members present, JSON fields and shapes, the generator's visibility guarantees
(synthetic-data/docs/data_format.md section 6), visible_px equal to the mask pixel counts, and
the exported per-frame poses equal to our composition of the world poses. On a seeded subset,
the projected voxels must cover the visible masks (a coarse convention check).
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from omegaconf import DictConfig

from vidsam3d.data import reader
from vidsam3d.data.samples import VAL, split_folder, split_objects
from vidsam3d.utils import setup_logging

log = logging.getLogger("vidsam3d")

MIN_VISIBLE_FRAC = 0.1          # >= 10 % of the amodal area visible
MIN_VISIBLE_IMAGE_FRAC = 0.002  # >= 0.2 % of the image (524 px at 512^2)
MIN_FRAMES = 32                 # frame subset S = {o, o+2, ..., o+30}, o in {0, 1}
MIN_VOXEL_COVERAGE = 0.9


def check_clip(clip: reader.Clip, voxels: dict[str, np.ndarray], frames: list[int]) -> list[str]:
    """Problems found in one clip (empty list if it is fine)."""
    info, errors = clip.info, []
    T, N, (H, W) = clip.num_frames, clip.num_objects, clip.image_size
    if T < MIN_FRAMES:
        errors.append(f"{T} frames, need >= {MIN_FRAMES}")
    if info["num_objects"] != N:
        errors.append(f"num_objects {info['num_objects']} != {N} objects")
    expected = [f"{kind}.{t:03d}.{ext}" for kind, ext in (("rgb", "jpg"), ("visible", "png")) for t in range(T)]
    missing = [e for e in expected if e not in clip.ref.members]
    if missing:
        return errors + [f"{len(missing)} missing members, e.g. {missing[0]}"]
    for k in range(1, N + 1):
        o = reader.object_entry(clip, k)
        packed = voxels.get(o["asset_id"])
        if packed is None or packed.dtype != np.uint8 or packed.shape != (reader.VOXEL_RES**3 // 8,):
            errors.append(f"object {k}: asset {o['asset_id']} missing from the voxels file or not packed 64^3")
        for key, shape in (("R_co", (T, 3, 3)), ("t_co", (T, 3)), ("s_co", (T, 3))):
            if np.shape(o[key]) != shape:
                errors.append(f"object {k}: {key} shape {np.shape(o[key])}, expected {shape}")
        if errors:
            continue
        for t in range(T):
            ours = reader.camera_frame_pose(clip, k, t)
            theirs = (np.array(o["R_co"][t]), np.array(o["t_co"][t]), np.array(o["s_co"][t]))
            if not all(np.allclose(a, b, atol=1e-6) for a, b in zip(ours, theirs)):
                errors.append(f"object {k} t={t}: exported pose differs from R_cw R_wo, R_cw t_wo + t_cw")
                break
    vis = reader.visibility(clip)
    px = reader.visibility(clip, "visible_px")
    if vis.shape != (T, N) or px.shape != (T, N):
        return errors + [f"visibility shape {vis.shape}, expected {(T, N)}"]
    if vis.min() < MIN_VISIBLE_FRAC:
        errors.append(f"visible_frac {vis.min():.3f} < {MIN_VISIBLE_FRAC}")
    if px.min() < MIN_VISIBLE_IMAGE_FRAC * H * W:
        errors.append(f"visible_px below {MIN_VISIBLE_IMAGE_FRAC:.1%} of the image")
    for t in frames:
        rgb, inst = reader.load_rgb(clip, t), reader.load_instance_mask(clip, t)
        if rgb.shape != (H, W, 3):
            errors.append(f"rgb.{t:03d}: shape {rgb.shape}, expected {(H, W, 3)}")
        if inst.dtype != np.uint8 or inst.shape != (H, W) or inst.max() > N:
            errors.append(f"visible.{t:03d}: {inst.dtype} {inst.shape}, max {inst.max()} (N={N})")
            continue
        counts = [int((inst == k).sum()) for k in range(1, N + 1)]
        if counts != px[t].tolist():
            errors.append(f"visible.{t:03d}: pixel counts {counts} != visible_px {px[t].tolist()}")
    return errors


def validate_data(cfg: DictConfig, split: str, num_projection_clips: int = 20) -> dict:
    """Check every clip of an export folder (val is checked as part of train)."""
    setup_logging()
    folder = split_folder(cfg.data, split)
    data = reader.open_split(Path(cfg.paths.data_root) / folder, str(cfg.paths.cache_root))
    table = reader.voxel_table(str(data.voxels_file))
    rng = np.random.default_rng(int(cfg.seed))
    clip_ids = list(data.clips)
    projection_ids = set(rng.choice(clip_ids, size=min(num_projection_clips, len(clip_ids)), replace=False))
    problems, coverage = {}, []
    for clip_id in clip_ids:
        try:
            clip = data.load_clip(clip_id)
            T = clip.num_frames
            errors = check_clip(clip, table, sorted({0, T // 2, T - 1}))
            if not errors and clip_id in projection_ids:
                for k in range(1, clip.num_objects + 1):
                    vox = reader.load_voxels(data.voxels_file, reader.object_entry(clip, k)["asset_id"])
                    frac = reader.voxel_coverage(clip, k, T // 2, vox)
                    coverage.append(frac)
                    if frac < MIN_VOXEL_COVERAGE:
                        errors.append(f"object {k}: projected voxels cover only {frac:.2f} of the visible mask")
        except Exception as ex:                   # unreadable member, JSON or image
            errors = [f"{type(ex).__name__}: {ex}"]
        if errors:
            problems[clip_id] = errors
    clip_assets = {cid: ref.asset_ids for cid, ref in data.clips.items()}
    report = {"split": split, "folder": folder, "clips": len(clip_ids), "assets": len(table),
              "clips_with_problems": len(problems), "coverage_checked_objects": len(coverage),
              "coverage_min": float(min(coverage)) if coverage else None, "problems": problems}
    if folder == cfg.data.train_split:
        val = split_objects(clip_assets, VAL, cfg.data)
        report["val"] = {"clips": len(val), "objects": sum(map(len, val.values())),
                         "train_clips": len(clip_ids) - len(val)}
    for clip_id, errors in list(problems.items())[:20]:
        log.error("%s: %s", clip_id, "; ".join(errors))
    log.info("validate-data %s: %d clips, %d with problems, voxel coverage min %s%s", folder, len(clip_ids),
             len(problems), report["coverage_min"], f", val {report['val']}" if "val" in report else "")
    return report
