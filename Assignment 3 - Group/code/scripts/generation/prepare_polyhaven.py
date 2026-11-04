"""Download CC0 Poly Haven panoramas and prepare frozen, source-disjoint background pools.

Run once on a CPU node from Assignment 3 - Group/code/: uv run python scripts/generation/prepare_polyhaven.py
Downloads are cached. Re-running identical arguments resumes; changed arguments require new
output directories. Uses the public API, not website scraping. Powered by Poly Haven.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import re

import numpy as np
from PIL import Image
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

API = "https://api.polyhaven.com"
LICENSE = "https://creativecommons.org/publicdomain/zero/1.0/"
DEFAULT_ROOT = Path(__file__).resolve().parents[2]


def digest(path: Path, algorithm: str = "sha256") -> str:
    h = hashlib.new(algorithm)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: Path, value) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def source_split(asset_id: str, eval_frac: float) -> str:
    value = int(hashlib.sha256(asset_id.encode()).hexdigest()[:16], 16) / 2**64
    return "eval" if value < eval_frac else "train"


def perspective(panorama: np.ndarray, yaw_deg: float, width: int, height: int, hfov: float) -> np.ndarray:
    """Level rectilinear view of an equirectangular RGB panorama, integer pixel centers.

    Longitude wraps at the panorama seam. Latitude clamps at the poles. Work in row chunks
    to bound RAM use on CPU nodes. JPEG inputs are already tone-mapped by Poly Haven.
    """
    ph, pw = panorama.shape[:2]
    focal = width / (2 * np.tan(np.deg2rad(hfov) / 2))
    x = (np.arange(width) - (width - 1) / 2) / focal
    out = np.empty((height, width, 3), np.uint8)
    for start in range(0, height, 128):
        stop = min(start + 128, height)
        y = ((np.arange(start, stop) - (height - 1) / 2) / focal)[:, None]
        lon = np.arctan2(x, 1)[None, :] + np.deg2rad(yaw_deg)
        lat = np.arctan2(-y, np.sqrt(1 + x[None, :] ** 2))
        u = np.broadcast_to(((lon / (2 * np.pi) + 0.5) * pw - 0.5) % pw, lat.shape)
        v = np.clip((0.5 - lat / np.pi) * ph - 0.5, 0, ph - 1)
        ix, iy = np.floor(u).astype(int), np.floor(v).astype(int)
        ax, ay = (u - ix)[..., None], (v - iy)[..., None]
        jx, jy = (ix + 1) % pw, np.minimum(iy + 1, ph - 1)
        pixels = ((1 - ay) * ((1 - ax) * panorama[iy, ix] + ax * panorama[iy, jx])
                  + ay * ((1 - ax) * panorama[jy, ix] + ax * panorama[jy, jx]))
        out[start:stop] = np.rint(pixels).clip(0, 255).astype(np.uint8)
    return out


def download(session, file_info: dict, dest: Path) -> None:
    if dest.exists() and digest(dest, "md5") == file_info["md5"].lower():
        return
    tmp = dest.with_suffix(".part")
    with session.get(file_info["url"], stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with tmp.open("wb") as f:
            for block in response.iter_content(1 << 20):
                f.write(block)
    if digest(tmp, "md5") != file_info["md5"].lower():
        tmp.unlink()
        raise ValueError(f"Download checksum mismatch: {dest.name}")
    tmp.replace(dest)


def prepare(args) -> dict:
    root = args.root.resolve()
    raw, out = root / args.raw, root / args.out
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    parameters = {k: getattr(args, k) for k in ("limit", "seed", "views", "width", "height", "hfov", "eval_frac")}
    with requests.Session() as session:
        session.headers["User-Agent"] = "sam3dvid-polyhaven-backgrounds/1.0 (ATDL research dataset)"
        session.mount("https://", HTTPAdapter(max_retries=Retry(
            total=5, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])))
        selection_path = raw / "selection.json"
        if selection_path.exists():
            selection = json.loads(selection_path.read_text())
            if selection["parameters"] != parameters:
                raise ValueError("Preparation settings changed; use new --raw and --out directories.")
        else:
            response = session.get(f"{API}/assets", params={"type": "hdris"}, timeout=60)
            response.raise_for_status()
            catalog = response.json()
            candidates = sorted(catalog, key=lambda aid: hashlib.sha256(f"{args.seed}:{aid}".encode()).hexdigest())
            if len(candidates) < args.limit:
                raise ValueError(f"Only {len(candidates)} HDRIs available; requested {args.limit}.")
            selected = candidates[:args.limit]
            if any(not re.fullmatch(r"[a-zA-Z0-9_-]+", aid) for aid in selected):
                raise ValueError("Unexpected asset ID in API response")
            selection = {"parameters": parameters, "assets": {aid: catalog[aid] for aid in selected}}
            atomic_json(selection_path, selection)
        manifest_path = out / "manifest.json"
        if manifest_path.exists():
            previous = json.loads(manifest_path.read_text())
            if previous["parameters"] != parameters or previous["asset_ids"] != sorted(selection["assets"]):
                raise ValueError("Output pool belongs to another selection; use a new --out directory.")
        records = []
        for aid, info in sorted(selection["assets"].items()):
            print(f"Preparing Poly Haven: {aid}", flush=True)
            files_path = raw / f"{aid}.files.json"
            if not files_path.exists():
                response = session.get(f"{API}/files/{aid}", timeout=60)
                response.raise_for_status()
                atomic_json(files_path, response.json())
            files = json.loads(files_path.read_text())
            file_info = files.get("tonemapped")
            if not file_info or not file_info.get("md5"):
                raise ValueError(f"{aid}: no checksummed tone-mapped panorama; pool not published")
            jpg = raw / f"{aid}.jpg"
            download(session, file_info, jpg)
            with Image.open(jpg) as im:
                if im.width != 2 * im.height:
                    raise ValueError(f"{aid}: panorama must be 2:1, got {im.size}")
                panorama = np.asarray(im.convert("RGB"))
            source_hash = digest(jpg)
            for view in range(args.views):
                image_id = f"polyhaven_{aid}_v{view:02d}"
                path = raw / f"{image_id}.jpg"
                yaw = view * 360 / args.views
                if not path.exists():
                    pixels = perspective(panorama, yaw, args.width, args.height, args.hfov)
                    tmp = path.with_suffix(".tmp")
                    Image.fromarray(pixels).save(tmp, format="JPEG", quality=95, subsampling=0)
                    tmp.replace(path)
                with Image.open(path) as im:
                    im.load()
                    if im.size != (args.width, args.height):
                        raise ValueError(f"Unexpected dimensions in cached view {path}")
                records.append({"image_id": image_id, "path": path.relative_to(root).as_posix(),
                                "width": args.width, "height": args.height,
                                "source": "polyhaven", "source_id": aid,
                                "source_url": f"https://polyhaven.com/a/{aid}",
                                "download_url": file_info["url"], "authors": info.get("authors", {}),
                                "license": "CC0-1.0", "license_url": LICENSE,
                                "source_sha256": source_hash, "sha256": digest(path),
                                "yaw_deg": yaw, "perspective_hfov_deg": args.hfov,
                                "split": source_split(aid, args.eval_frac)})
        pools = {split: [r for r in records if r["split"] == split] for split in ("train", "eval")}
        if any(not pool for pool in pools.values()):
            raise ValueError("An empty split was selected; increase --limit using new output directories.")
        manifest = {"source": "Poly Haven", "license": "CC0-1.0", "license_url": LICENSE,
                    "parameters": parameters, "asset_ids": sorted(selection["assets"]), "images": records}
        if manifest_path.exists() and json.loads(manifest_path.read_text()) != manifest:
            raise ValueError("Frozen background manifest changed; use new output directories.")
        atomic_json(manifest_path, manifest)
        for split, pool in pools.items():
            path = out / f"{split}.jsonl"
            tmp = path.with_suffix(".tmp")
            tmp.write_text("".join(json.dumps(r) + "\n" for r in pool))
            tmp.replace(path)
        stats = {"source": "polyhaven", "source_assets": len(selection["assets"]),
                 "pools": {split: len(pool) for split, pool in pools.items()}}
        atomic_json(out / "stats.json", stats)
        return stats


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--raw", type=Path, default=Path("data/raw/polyhaven"))
    parser.add_argument("--out", type=Path, default=Path("data/bg_pools_polyhaven"))
    parser.add_argument("--limit", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--views", type=int, default=4)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--hfov", type=float, default=130)
    parser.add_argument("--eval-frac", type=float, default=0.2)
    args = parser.parse_args(argv)
    if args.limit < 2 or args.views < 1 or args.height < 1024 or args.width < args.height:
        parser.error("Need >=2 panoramas, >=1 view, and landscape images with short side >=1024")
    if not 0 < args.hfov < 180 or not 0 < args.eval_frac < 1:
        parser.error("Need 0 < hfov < 180 and 0 < eval-frac < 1")
    for path in (args.raw, args.out):
        if path.is_absolute() or ".." in path.parts:
            parser.error("--raw and --out must be relative paths inside --root")
    try:
        print(json.dumps(prepare(args), indent=2))
    except (ValueError, requests.RequestException) as error:
        parser.exit(1, f"Background preparation failed: {error}\n")


if __name__ == "__main__":
    main()
