"""Clip generation: seed -> clip folder (spec Sections 4.3-4.7 and 5).

Usage:
    uv run python -m sam3dvid.generate --config configs/m0.yaml --split eval --seeds 0 [1 2 ...] [--overwrite]

Existing clips are skipped unless --overwrite is given; rejected or crashed seeds are appended
to <dataset_root>/failures.jsonl and the run continues.
"""

from __future__ import annotations

import argparse
import fcntl
import json
import shutil
import subprocess
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np
import yaml

from . import __version__
from .background import background_K, covers, homography, relative_rotation, warp
from .blenderrun import run_blenderproc
from .clipio import write_clip
from .config import Config
from .conventions import blender_cam_matrix_from_T_cw, intrinsics, object_matrix
from .precheck import estimate_visibility, voxel_centers
from .sampling import (
    PlacementError,
    resample_arc,
    sample_layout,
    sample_light,
    sample_num_objects,
    sample_trajectory,
    scene_extent,
)
from .validate import visibility_failures, visibility_stats

SOURCE_FOR_SPLIT = {"train": "objaverse", "eval": "gso"}


class ClipRejected(RuntimeError):
    pass


def git_commit() -> str | None:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True).stdout.strip()
        return out.stdout.strip() + ("-dirty" if dirty else "")
    except subprocess.CalledProcessError:
        return None


def asset_pool(cfg: Config, split: str) -> list[str]:
    root = Path(cfg.dataset_root) / "assets" / split
    return sorted(p.name for p in root.iterdir() if (p / "meta.json").exists())


def background_pool(cfg: Config, split: str) -> list[dict]:
    path = Path(cfg.bg_pool_dir) / f"{split}.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def rng_for(seed: int, *stream: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence([seed, *stream])))


def plan_attempt(cfg: Config, rng: np.random.Generator, asset_ids: list[str], centers: list[np.ndarray], bg_size: tuple[int, int]):
    """Layout, trajectory and arc resampling until the background covers every frame."""
    objects = sample_layout(rng, cfg, asset_ids)
    center, radius = scene_extent(objects, centers)
    traj = sample_trajectory(rng, cfg, center, radius)
    K = intrinsics(cfg.width, cfg.height, cfg.hfov_deg)
    K_bg = background_K(*bg_size, cfg.bg_hfov_deg)
    for arc_retries in range(cfg.max_arc_retries + 1):
        frames, ref = traj.frames(), traj.reference()
        if all(
            covers(K, relative_rotation(T, ref), K_bg, *bg_size, cfg.width, cfg.height, cfg.bg_margin_px)
            for T in frames
        ):
            return objects, traj, frames, ref, arc_retries
        resample_arc(rng, cfg, traj)
    raise ClipRejected(f"background coverage failed after {cfg.max_arc_retries} arc resamples")


def precheck(cfg: Config, K, frames, objects, centers) -> None:
    for t, T_cw in enumerate(frames):
        vis, amo = estimate_visibility(K, T_cw, objects, centers, cfg.width, cfg.height, cfg.precheck_resolution)
        stats = {
            "visible_px": vis[None],
            "amodal_px": amo[None],
            "visible_frac": np.divide(vis, amo, out=np.zeros_like(vis), where=amo > 0)[None],
        }
        failures = visibility_failures(stats, cfg, slack=cfg.precheck_slack)
        if failures:
            raise ClipRejected(f"precheck t={t}: {failures[0].split(' ', 1)[1]}")


def render(cfg: Config, scene: dict, workdir: Path) -> dict[str, np.ndarray]:
    scene_path, out_path = workdir / "scene.json", workdir / "render.npz"
    scene_path.write_text(json.dumps(scene))
    run_blenderproc(
        "render_clip.py", str(scene_path), str(out_path),
        install_path=cfg.blender_install_path, log_path=workdir / "blender.log",
    )
    with np.load(out_path) as data:
        return {k: data[k] for k in data.files}


def composite(rgba: np.ndarray, background: np.ndarray) -> np.ndarray:
    alpha = rgba[..., 3:4].astype(np.float32) / 255
    out = rgba[..., :3].astype(np.float32) * alpha + background.astype(np.float32) * (1 - alpha)
    return np.clip(np.round(out), 0, 255).astype(np.uint8)


def generate_clip(cfg: Config, split: str, seed: int, out_root: Path) -> dict:
    """Generate, validate and write one clip; returns its manifest entry."""
    t_start = time.time()
    clip_id = f"{seed:08d}"
    dataset_root = Path(cfg.dataset_root)
    assets, bgs = asset_pool(cfg, split), background_pool(cfg, split)

    # Object count, assets, background and light are fixed per seed; retries only redraw poses
    # and camera, so rejections don't bias N, the asset mix or the backgrounds.
    base = rng_for(seed, 0)
    n = sample_num_objects(base, cfg)
    asset_ids = [assets[i] for i in base.choice(len(assets), size=n, replace=False)]
    bg = bgs[int(base.integers(len(bgs)))]
    light = sample_light(base, cfg)

    K = intrinsics(cfg.width, cfg.height, cfg.hfov_deg)
    centers = [voxel_centers(np.load(dataset_root / "assets" / split / a / "voxels.npy")) for a in asset_ids]
    bg_image = cv2.cvtColor(cv2.imread(bg["path"], cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    bg_size = (bg_image.shape[1], bg_image.shape[0])
    K_bg = background_K(*bg_size, cfg.bg_hfov_deg)

    rejections, render_time = [], 0.0
    for attempt in range(cfg.max_retries):
        rng = rng_for(seed, 1, attempt)
        try:
            objects, traj, frames, ref, arc_retries = plan_attempt(cfg, rng, asset_ids, centers, bg_size)
            precheck(cfg, K, frames, objects, centers)
        except (PlacementError, ClipRejected) as e:
            rejections.append({"attempt": attempt, "stage": "plan", "reason": str(e)})
            continue

        scene = {
            "render_seed": int(rng_for(seed, 2, attempt).integers(2**31)),
            "samples": cfg.cycles_samples,
            "denoiser": cfg.denoiser,
            "light": light,
            "camera": {
                "K": K.tolist(), "width": cfg.width, "height": cfg.height,
                "blender_cam2world": [blender_cam_matrix_from_T_cw(T).tolist() for T in frames],
            },
            "objects": [
                {
                    "k": k + 1,
                    "mesh_path": str((dataset_root / "assets" / split / o.asset_id / "mesh.glb").resolve()),
                    "matrix_world": object_matrix(o.R_wo, o.t_wo, o.s_wo).tolist(),
                }
                for k, o in enumerate(objects)
            ],
        }
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.time()
            out = render(cfg, scene, Path(tmp))
            render_time += time.time() - t0

        stats = visibility_stats(out["visible"], out["amodal"])
        failures = visibility_failures(stats, cfg)
        if failures:
            rejections.append({"attempt": attempt, "stage": "render", "reason": f"{len(failures)} violations, first: {failures[0]}"})
            continue

        backgrounds, defined = zip(*(warp(bg_image, homography(K, relative_rotation(T, ref), K_bg), cfg.width, cfg.height) for T in frames))
        if not all(d.all() for d in defined):
            rejections.append({"attempt": attempt, "stage": "composite", "reason": "undefined background pixels"})
            continue

        depth = np.where(out["visible"] > 0, out["depth"], np.nan).astype(np.float32)
        rgb = np.stack([composite(fg, b) for fg, b in zip(out["colors"], backgrounds)])
        cameras = {
            "convention": "opencv", "width": cfg.width, "height": cfg.height, "K": K.tolist(),
            "frames": [{"t": t, "T_cw": T.tolist()} for t, T in enumerate(frames)],
        }
        objects_json = [
            {
                "k": k + 1,
                "asset_id": o.asset_id,
                "source": SOURCE_FOR_SPLIT[split],
                "mesh": f"assets/{split}/{o.asset_id}/mesh.glb",
                "voxels": f"assets/{split}/{o.asset_id}/voxels.npy",
                "R_wo": o.R_wo.tolist(), "t_wo": o.t_wo.tolist(), "s_wo": o.s_wo.tolist(),
            }
            for k, o in enumerate(objects)
        ]
        meta = {
            "clip_id": clip_id, "split": split, "seed": seed, "attempt": attempt,
            "generator_version": __version__, "git_commit": git_commit(),
            "num_objects": n,
            "arc_span_deg": abs(traj.arc_span_deg), "arc_direction": int(np.sign(traj.arc_span_deg)),
            "start_azimuth_deg": traj.start_azimuth_deg, "start_elevation_deg": traj.start_elevation_deg,
            "elevation_drift_deg": traj.elevation_drift_deg, "orbit_radius": traj.orbit_radius,
            "scene_center": traj.center.tolist(), "hfov_deg": cfg.hfov_deg,
            "reference_T_cw": ref.tolist(),
            "light": light,
            "background": {"image_id": bg["image_id"], "width": bg_size[0], "height": bg_size[1],
                           "virtual_hfov_deg": cfg.bg_hfov_deg, "K_bg": K_bg.tolist()},
            "render": {"seed": scene["render_seed"], "samples": cfg.cycles_samples, "denoiser": cfg.denoiser},
            "visibility": {k: v.tolist() for k, v in stats.items()},
            "arc_retries": arc_retries,
            "rejections": rejections,
            "timing": {"render_s": round(render_time, 2), "total_s": round(time.time() - t_start, 2)},
        }
        clip_dir = out_root / "clips" / split / clip_id
        tmp_dir = clip_dir.with_name(clip_id + ".tmp")
        shutil.rmtree(tmp_dir, ignore_errors=True)
        write_clip(tmp_dir, rgb, out["colors"], out["visible"], out["amodal"], depth, cameras, objects_json, meta)
        shutil.rmtree(clip_dir, ignore_errors=True)
        tmp_dir.rename(clip_dir)
        return {
            "clip_id": clip_id, "split": split, "N": n, "arc_span_deg": abs(traj.arc_span_deg),
            "seed": seed, "path": f"clips/{split}/{clip_id}",
        }
    raise ClipRejected(f"seed {seed}: all {cfg.max_retries} attempts rejected: {rejections}")


def update_manifest(out_root: Path, entry: dict) -> None:
    path = out_root / "manifest.jsonl"
    # Several workers may finish clips at once. flock is unreliable on NFS/Lustre; the
    # manifest can always be rebuilt from the clips' meta.json.
    with open(out_root / "manifest.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        entries = {}
        if path.exists():
            entries = {e["clip_id"]: e for e in map(json.loads, path.read_text().splitlines()) if e}
        entries[entry["clip_id"]] = entry
        tmp = path.with_suffix(".tmp")
        tmp.write_text("".join(json.dumps(e) + "\n" for _, e in sorted(entries.items())))
        tmp.replace(path)


def log_failure(out_root: Path, split: str, seed: int, error: Exception) -> None:
    line = json.dumps({"split": split, "seed": seed, "error": type(error).__name__, "message": str(error)})
    with open(out_root / "failures.jsonl", "a") as f:
        f.write(line + "\n")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--split", choices=["train", "eval"], required=True)
    ap.add_argument("--seeds", type=int, nargs="+", required=True)
    ap.add_argument("--overwrite", action="store_true", help="regenerate clips that already exist")
    args = ap.parse_args()
    cfg = Config.load(args.config)
    out_root = Path(cfg.dataset_root)
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "config.yaml").write_text(yaml.safe_dump(cfg.to_dict(), sort_keys=False))
    for seed in args.seeds:
        if not args.overwrite and (out_root / "clips" / args.split / f"{seed:08d}" / "meta.json").exists():
            continue
        try:
            entry = generate_clip(cfg, args.split, seed, out_root)
        except (ClipRejected, subprocess.CalledProcessError) as e:
            log_failure(out_root, args.split, seed, e)
            print(f"FAILED seed {seed}: {type(e).__name__}", flush=True)
            continue
        update_manifest(out_root, entry)
        print(json.dumps(entry), flush=True)


if __name__ == "__main__":
    main()
