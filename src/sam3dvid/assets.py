"""Asset preprocessing (spec Section 4.1): normalized mesh.glb, 64^3 voxels.npy and meta.json.

Usage:
    uv run python -m sam3dvid.assets --source gso --split eval [--limit 50]
    uv run python -m sam3dvid.assets --source objaverse --split train [--limit 50] [--workers 8]

Sources read `--raw` (default data/raw/<source>), as filled by scripts/download_gso.py and
scripts/download_objaverse.py. Finished assets (meta.json present) are skipped; assets that crash,
hang or fail in Blender are logged to <dataset_root>/assets/failed_<split>.jsonl and skipped until
`--retry-failed`. `--refresh-meta` rewrites the source fields (license, captions, ...) of finished
assets from the current index without reprocessing them.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import open3d as o3d
import trimesh
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from .blenderrun import run_blenderproc
from .config import Config

GRID = 64

# Sketchfab's license codes (Objaverse 1.0 metadata) as SPDX identifiers; all are CC 4.0.
SKETCHFAB_LICENSES = {
    "by": "CC-BY-4.0",
    "by-sa": "CC-BY-SA-4.0",
    "by-nd": "CC-BY-ND-4.0",
    "by-nc": "CC-BY-NC-4.0",
    "by-nc-sa": "CC-BY-NC-SA-4.0",
    "by-nc-nd": "CC-BY-NC-ND-4.0",
    "cc0": "CC0-1.0",
}


@dataclass
class Asset:
    asset_id: str
    # Called with an empty directory it may use; returns the mesh file Blender imports.
    stage: Callable[[Path], Path]
    # Source-specific meta.json fields (source, source_name, license, ...).
    meta: dict


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
    mesh.merge_vertices()
    extent = mesh.bounds[1] - mesh.bounds[0]
    areas = mesh.area_faces
    n = mesh.face_normals
    moment = (n * areas[:, None]).T @ n / areas.sum()
    # Vertex-connected components via scipy: trimesh's mesh.split needs networkx, which isn't a
    # dependency, and is slow on scenes with thousands of parts.
    f = mesh.faces
    graph = coo_matrix((np.ones(3 * len(f)), (f.ravel(), np.roll(f, -1, axis=1).ravel())), shape=(len(mesh.vertices),) * 2)
    num_parts, vertex_label = connected_components(graph, directed=False)
    label = vertex_label[f[:, 0]]
    part_area = np.bincount(label, weights=areas, minlength=num_parts)
    centers = np.stack([np.bincount(label, weights=areas * mesh.triangles_center[:, k], minlength=num_parts) for k in range(3)], 1)
    centers = centers / np.maximum(part_area, 1e-30)[:, None]
    used = part_area > 0
    main = np.argmax(part_area)
    far = part_area[used & (np.linalg.norm(centers - centers[main], axis=1) > 0.25)].sum()
    return {
        "extent_min": float(extent.min()),
        "normal_eig_min": float(np.linalg.eigvalsh(moment)[0]),
        "outlier_area_frac": float(far / mesh.area),
        "num_components": int(used.sum()),
    }


def asset_id_for(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_-]", "_", name)


def stage_gso(model_dir: Path, staging: Path) -> Path:
    """Copy OBJ, MTL and texture side by side; GSO's MTL names the texture without its folder."""
    staging.mkdir(parents=True, exist_ok=True)
    shutil.copy(model_dir / "meshes" / "model.obj", staging)
    shutil.copy(model_dir / "meshes" / "model.mtl", staging)
    shutil.copy(model_dir / "materials" / "textures" / "texture.png", staging)
    return staging / "model.obj"


def gso_assets(raw: Path, limit: int | None) -> list[Asset]:
    models = sorted(p for p in raw.iterdir() if (p / ".done").exists())
    models = models[:limit] if limit else models
    return [
        Asset(
            asset_id_for(m.name),
            lambda staging, m=m: stage_gso(m, staging),
            {"source": "gso", "source_name": m.name, "license": "CC-BY-4.0"},
        )
        for m in models
    ]


def objaverse_assets(raw: Path, limit: int | None) -> list[Asset]:
    """Downloaded GLBs in the order of the download script's seeded selection, so `limit` takes a prefix of it."""
    index = json.loads((raw / "index.json").read_text())
    uids = [u for u in index["selection"] if (raw / "glbs" / f"{u}.glb").exists()]
    uids = uids[:limit] if limit else uids
    assets = []
    for uid in uids:
        o = index["objects"][uid]
        meta = {
            "source": "objaverse",
            "source_name": uid,
            "license": SKETCHFAB_LICENSES.get(o["license"], o["license"]),
            "license_code": o["license"],
            "source_url": o["source_url"],
            "sha256": o["sha256"],
            "aesthetic_score": o["aesthetic_score"],
            "name": o["name"],
            "author": o["author"],
            "captions": o.get("captions", []),
        }
        # GLBs are self-contained, so Blender reads them in place.
        assets.append(Asset(uid, lambda staging, p=(raw / "glbs" / f"{uid}.glb").resolve(): p, meta))
    return assets


def failed_path(cfg: Config, split: str) -> Path:
    return Path(cfg.dataset_root) / "assets" / f"failed_{split}.jsonl"


def load_failed(path: Path) -> dict[str, str]:
    if not path.exists():
        return {}
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["asset_id"]: r["reason"] for r in rows}


class FailureLog:
    """Append-only jsonl of assets that could not be preprocessed; safe to share between threads."""

    def __init__(self, path: Path):
        self.path = path
        self.lock = threading.Lock()

    def add(self, asset_id: str, reason: str) -> None:
        print(f"FAILED {asset_id}: {reason}", flush=True)
        with self.lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a") as f:
                f.write(json.dumps({"asset_id": asset_id, "reason": reason}) + "\n")


def preprocess(
    cfg: Config, assets: list[Asset], split: str, chunk: int = 50, workers: int = 1,
    stall_timeout: float | None = 600, retry_failed: bool = False,
) -> None:
    out_root = Path(cfg.dataset_root) / "assets" / split
    failures = FailureLog(failed_path(cfg, split))
    failed = {} if retry_failed else load_failed(failures.path)
    todo = [a for a in assets if not (out_root / a.asset_id / "meta.json").exists() and a.asset_id not in failed]
    print(f"{len(todo)} of {len(assets)} assets to process ({len(failed)} previously failed)", flush=True)
    t0 = time.monotonic()
    chunks = [todo[i : i + chunk] for i in range(0, len(todo), chunk)]
    with ThreadPoolExecutor(workers) as pool:
        for fut in [pool.submit(preprocess_chunk, cfg, c, out_root, failures, stall_timeout) for c in chunks]:
            fut.result()
    ok = sum((out_root / a.asset_id / "meta.json").exists() for a in todo)
    dt = time.monotonic() - t0
    print(f"{ok} of {len(todo)} assets done in {dt:.0f} s ({dt / max(len(todo), 1):.1f} s per asset)", flush=True)


def refresh_meta(cfg: Config, assets: list[Asset], split: str) -> None:
    out_root = Path(cfg.dataset_root) / "assets" / split
    n = 0
    for a in assets:
        path = out_root / a.asset_id / "meta.json"
        if path.exists():
            path.write_text(json.dumps({**json.loads(path.read_text()), **a.meta}, indent=1))
            n += 1
    print(f"refreshed {n} of {len(assets)} meta.json files", flush=True)


def preprocess_chunk(
    cfg: Config, assets: list[Asset], out_root: Path, failures: FailureLog, stall_timeout: float | None
) -> None:
    """Run Blender over `assets`, then voxelize and score what it finished.

    Blender runs the jobs in order, so when it crashes or stalls the first job without output is the
    culprit: it is logged as failed and Blender restarts on the jobs after it.
    """
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        pending = list(assets)
        for a in pending:
            shutil.rmtree(out_root / a.asset_id, ignore_errors=True)
        sources = {a.asset_id: a.stage(tmp / "stage" / asset_id_for(a.asset_id)) for a in assets}
        while pending:
            jobs = [{"source": str(sources[a.asset_id]), "out_dir": str((out_root / a.asset_id).resolve())} for a in pending]
            jobs_path = tmp / "jobs.json"
            jobs_path.write_text(json.dumps(jobs))
            error = None
            try:
                run_blenderproc(
                    "normalize_assets.py", str(jobs_path),
                    install_path=cfg.blender_install_path, custom_path=cfg.blender_custom_path,
                    stall_timeout=stall_timeout,
                )
            except subprocess.CalledProcessError as e:
                error = f"Blender crashed (exit {e.returncode})"
            except subprocess.TimeoutExpired:
                error = f"Blender stalled for {stall_timeout:.0f} s"
            unfinished = []
            for a in pending:
                d = out_root / a.asset_id
                if (d / "normalize.json").exists():
                    continue
                if (d / "error.txt").exists():
                    failures.add(a.asset_id, (d / "error.txt").read_text())
                    shutil.rmtree(d, ignore_errors=True)
                else:
                    unfinished.append(a)
            if error is None:
                for a in unfinished:
                    failures.add(a.asset_id, "no Blender output")
                break
            if unfinished:
                failures.add(unfinished[0].asset_id, error)
                shutil.rmtree(out_root / unfinished[0].asset_id, ignore_errors=True)
            pending = unfinished[1:]

    for a in assets:
        out = out_root / a.asset_id
        if (out / "normalize.json").exists():
            try:
                postprocess(a, out)
            except Exception as e:
                failures.add(a.asset_id, f"{type(e).__name__}: {e}")
                shutil.rmtree(out, ignore_errors=True)


def postprocess(asset: Asset, out: Path) -> None:
    indices = trellis_voxelize(out / "mesh.ply")
    voxels = np.zeros((GRID,) * 3, bool)
    voxels[tuple(indices.T)] = True
    np.save(out / "voxels.npy", voxels)
    mesh = trimesh.load(out / "mesh.ply", force="mesh", process=False)
    meta = {
        "asset_id": out.name,
        **asset.meta,
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
    ap.add_argument("--source", choices=["gso", "objaverse"], required=True)
    ap.add_argument("--raw", type=Path, default=None, help="default: data/raw/<source>")
    ap.add_argument("--split", choices=["train", "eval"], default=None, help="default: eval for gso, train for objaverse")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=None, help="assets per Blender process; default 50 (gso), 20 (objaverse)")
    ap.add_argument("--workers", type=int, default=1, help="parallel Blender processes")
    ap.add_argument("--stall-timeout", type=float, default=600, help="seconds without a finished asset before Blender is killed")
    ap.add_argument("--retry-failed", action="store_true")
    ap.add_argument("--refresh-meta", action="store_true")
    args = ap.parse_args()
    raw = args.raw or Path("data/raw") / args.source
    split = args.split or ("eval" if args.source == "gso" else "train")
    assets = (gso_assets if args.source == "gso" else objaverse_assets)(raw, args.limit)
    if args.refresh_meta:
        refresh_meta(Config.load(args.config), assets, split)
        return
    preprocess(
        Config.load(args.config), assets, split, chunk=args.chunk or (50 if args.source == "gso" else 20),
        workers=args.workers, stall_timeout=args.stall_timeout, retry_failed=args.retry_failed,
    )


if __name__ == "__main__":
    main()
