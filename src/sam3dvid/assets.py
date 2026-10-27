"""Asset preprocessing (spec Section 4.1): normalized mesh.glb, 64^3 voxels.npy and meta.json.

Usage:
    uv run python -m sam3dvid.assets --source gso --raw data/raw/gso --split eval [--limit 50]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh

from .blenderrun import run_blenderproc
from .config import Config

GRID = 64


def trellis_voxelize(ply_path: Path) -> np.ndarray:
    """Occupied voxel indices, replicating TRELLIS dataset_toolkits/voxelize.py exactly."""
    mesh = o3d.io.read_triangle_mesh(str(ply_path))
    vertices = np.clip(np.asarray(mesh.vertices), -0.5 + 1e-6, 0.5 - 1e-6)
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    voxel_grid = o3d.geometry.VoxelGrid.create_from_triangle_mesh_within_bounds(
        mesh, voxel_size=1 / 64, min_bound=(-0.5, -0.5, -0.5), max_bound=(0.5, 0.5, 0.5)
    )
    indices = np.array([voxel.grid_index for voxel in voxel_grid.get_voxels()])
    assert np.all(indices >= 0) and np.all(indices < 64), "Some vertices are out of bounds"
    return indices


def geometric_scores(mesh: trimesh.Trimesh) -> dict[str, float]:
    """Scores behind the App. B.1 geometric filter; thresholds are applied elsewhere.

    - extent_min: smallest side of the normalized bbox (point-like or flat when small).
    - normal_eig_min: smallest eigenvalue of the area-weighted normal second-moment matrix
      (near 0 when all normals are (anti)parallel, i.e. a flat sheet).
    - outlier_area_frac: surface-area share of connected components whose centroid is more than
      0.25 from the area-weighted centroid of the largest component.
    """
    extent = mesh.bounds[1] - mesh.bounds[0]
    areas = mesh.area_faces
    n = mesh.face_normals
    moment = (n * areas[:, None]).T @ n / areas.sum()
    parts = mesh.split(only_watertight=False)
    main = max(parts, key=lambda p: p.area)
    main_c = main.triangles_center.T @ main.area_faces / main.area
    far = sum(p.area for p in parts if np.linalg.norm(p.triangles_center.T @ p.area_faces / p.area - main_c) > 0.25)
    return {
        "extent_min": float(extent.min()),
        "normal_eig_min": float(np.linalg.eigvalsh(moment)[0]),
        "outlier_area_frac": float(far / mesh.area),
        "num_components": len(parts),
    }


def asset_id_for(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def stage_gso(model_dir: Path, staging: Path) -> Path:
    """Copy OBJ, MTL and texture side by side; GSO's MTL names the texture without its folder."""
    staging.mkdir(parents=True)
    shutil.copy(model_dir / "meshes" / "model.obj", staging)
    shutil.copy(model_dir / "meshes" / "model.mtl", staging)
    shutil.copy(model_dir / "materials" / "textures" / "texture.png", staging)
    return staging / "model.obj"


def gso_models(raw: Path, limit: int | None) -> list[Path]:
    models = sorted(p for p in raw.iterdir() if (p / ".done").exists())
    return models[:limit] if limit else models


def preprocess_gso(cfg: Config, raw: Path, split: str, limit: int | None, chunk: int = 50) -> None:
    out_root = Path(cfg.dataset_root) / "assets" / split
    models = [m for m in gso_models(raw, limit) if not (out_root / asset_id_for(m.name) / "meta.json").exists()]
    print(f"{len(models)} GSO models to process", flush=True)
    for i in range(0, len(models), chunk):
        preprocess_gso_chunk(cfg, models[i : i + chunk], out_root)


def preprocess_gso_chunk(cfg: Config, models: list[Path], out_root: Path) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        jobs = [
            {"source": str(stage_gso(m, tmp / asset_id_for(m.name))), "out_dir": str((out_root / asset_id_for(m.name)).resolve())}
            for m in models
        ]
        (tmp / "jobs.json").write_text(json.dumps(jobs))
        try:
            run_blenderproc(
                "normalize_assets.py", str(tmp / "jobs.json"),
                install_path=cfg.blender_install_path, custom_path=cfg.blender_custom_path,
            )
        except subprocess.CalledProcessError as e:
            # Assets Blender finished before the crash are still post-processed below.
            print(f"FAILED Blender chunk ({e.returncode}); keeping finished assets", flush=True)

    for m in models:
        out = out_root / asset_id_for(m.name)
        if not (out / "normalize.json").exists():
            print(f"FAILED {m.name}: no Blender output", flush=True)
            continue
        try:
            postprocess_gso(m, out)
        except Exception as e:
            print(f"FAILED {m.name}: {type(e).__name__}: {e}", flush=True)


def postprocess_gso(m: Path, out: Path) -> None:
    indices = trellis_voxelize(out / "mesh.ply")
    voxels = np.zeros((GRID,) * 3, bool)
    voxels[tuple(indices.T)] = True
    np.save(out / "voxels.npy", voxels)
    mesh = trimesh.load(out / "mesh.ply", force="mesh", process=False)
    meta = {
        "asset_id": out.name,
        "source": "gso",
        "source_name": m.name,
        "license": "CC-BY-4.0",
        "normalization": json.loads((out / "normalize.json").read_text()),
        "num_voxels": int(voxels.sum()),
        "filter_scores": geometric_scores(mesh),
    }
    (out / "meta.json").write_text(json.dumps(meta, indent=1))
    (out / "normalize.json").unlink()
    (out / "mesh.ply").unlink()
    print(f"ok {out.name}: {meta['num_voxels']} voxels", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=None)
    ap.add_argument("--source", choices=["gso"], required=True)
    ap.add_argument("--raw", type=Path, default=Path("data/raw/gso"))
    ap.add_argument("--split", choices=["train", "eval"], default="eval")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    preprocess_gso(Config.load(args.config), args.raw, args.split, args.limit)


if __name__ == "__main__":
    main()
