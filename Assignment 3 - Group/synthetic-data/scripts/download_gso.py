"""Download Google Scanned Objects models from Gazebo Fuel and extract them.

Usage:
    uv run python scripts/download_gso.py [--out data/raw/gso] [--limit N]
        [--seed 0] [--workers 8] [--keep-zip]

With --limit, the subset is the first N names after sorting the catalogue and
shuffling it with --seed, so the same seed always yields the same models and a
larger limit extends a smaller one. Each model is extracted into <out>/<name>/
and marked complete with a .done file; complete models are skipped. Zips are
deleted after extraction unless --keep-zip is given. <out>/index.json lists the
full catalogue and which models are present locally.
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests
from tqdm import tqdm

log = logging.getLogger("download_gso")

API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
CHUNK = 1 << 20
DONE_MARKER = ".done"


def get_with_retry(url: str, retries: int = 6, timeout: float = 60.0, **kw) -> requests.Response:
    """GET url, retrying with exponential backoff on network errors and 5xx/429."""
    for attempt in range(1, retries + 1):
        try:
            r = requests.get(url, timeout=timeout, **kw)
            if r.status_code < 500 and r.status_code != 429:
                return r
            err = f"HTTP {r.status_code}"
            r.close()
        except requests.RequestException as e:
            err = type(e).__name__
        if attempt == retries:
            raise RuntimeError(f"GET {url} failed after {retries} attempts ({err})")
        wait = min(2**attempt, 60)
        log.warning("GET %s: attempt %d failed (%s), retrying in %ds", url, attempt, err, wait)
        time.sleep(wait)
    raise AssertionError("unreachable")


def list_models(per_page: int = 100) -> list[dict]:
    """Return the metadata of every model owned by GoogleResearch on Fuel."""
    models: list[dict] = []
    page = 1
    while True:
        r = get_with_retry(f"{API}?per_page={per_page}&page={page}")
        # Fuel answers 404 for a page past the end instead of an empty list.
        if r.status_code == 404:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        models.extend(batch)
        page += 1
    return models


def select(names: list[str], limit: int | None, seed: int) -> list[str]:
    """Return the first limit names of a seeded shuffle of the sorted names."""
    order = sorted(names)
    random.Random(seed).shuffle(order)
    return order if limit is None else order[:limit]


def download(name: str, dest: Path, retries: int = 6) -> None:
    """Download version 1 of model name as a zip to dest via a .part file."""
    q = quote(name, safe="")
    url = f"{API}/{q}/1/{q}.zip"
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            r = get_with_retry(url, stream=True)
            with r:
                if r.status_code != 200:
                    raise RuntimeError(f"HTTP {r.status_code}")
                total = int(r.headers.get("Content-Length", 0)) or None
                with part.open("wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            if total is not None and part.stat().st_size != total:
                raise RuntimeError("size mismatch")
            if not zipfile.is_zipfile(part):
                raise RuntimeError("not a zip file")
            part.replace(dest)
            return
        except (requests.RequestException, RuntimeError) as e:
            msg = str(e) if isinstance(e, RuntimeError) else type(e).__name__
            if attempt == retries:
                part.unlink(missing_ok=True)
                raise RuntimeError(f"{name}: download failed after {retries} attempts ({msg})") from None
            wait = min(2**attempt, 60)
            log.warning("%s: attempt %d failed (%s), retrying in %ds", name, attempt, msg, wait)
            time.sleep(wait)


def extract(zip_path: Path, dest_dir: Path) -> None:
    """Extract zip_path into dest_dir atomically and write the .done marker.

    Members that would land outside dest_dir are rejected.
    """
    tmp = dest_dir.with_name(dest_dir.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    root = tmp.resolve()
    with zipfile.ZipFile(zip_path) as zf:
        for m in zf.infolist():
            target = (tmp / m.filename).resolve()
            if not target.is_relative_to(root):
                raise RuntimeError(f"{zip_path.name}: unsafe member path {m.filename!r}")
        zf.extractall(tmp)
    (tmp / DONE_MARKER).write_text("")
    shutil.rmtree(dest_dir, ignore_errors=True)
    tmp.replace(dest_dir)


def process_model(name: str, out: Path, keep_zip: bool) -> bool:
    """Ensure model name is extracted under out. Returns False if it was already complete."""
    dest_dir = out / name
    if (dest_dir / DONE_MARKER).exists():
        return False
    zip_path = out / f"{name}.zip"
    if not zip_path.exists():
        download(name, zip_path)
    extract(zip_path, dest_dir)
    if not keep_zip:
        zip_path.unlink()
    return True


def write_index(out: Path, models: list[dict]) -> int:
    """Write out/index.json for the full catalogue. Returns the number downloaded."""
    names = sorted(m["name"] for m in models)
    downloaded = [n for n in names if (out / n / DONE_MARKER).exists()]
    index = {
        "source": API,
        "num_models": len(names),
        "num_downloaded": len(downloaded),
        "models": names,
        "downloaded": downloaded,
        "metadata": {m["name"]: m for m in sorted(models, key=lambda m: m["name"])},
    }
    tmp = out / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1) + "\n")
    tmp.replace(out / "index.json")
    return len(downloaded)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/raw/gso"))
    p.add_argument("--limit", type=int, help="number of models to download, default: all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--keep-zip", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    args.out.mkdir(parents=True, exist_ok=True)
    models = list_models()
    if not models:
        log.error("catalogue is empty")
        return 1
    names = [m["name"] for m in models]
    if len(set(names)) != len(names):
        log.error("catalogue contains duplicate names")
        return 1
    log.info("catalogue: %d models", len(names))
    write_index(args.out, models)

    chosen = select(names, args.limit, args.seed)
    failed: list[str] = []
    fetched = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(process_model, n, args.out, args.keep_zip): n for n in chosen}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="model"):
            name = futures[fut]
            try:
                fetched += fut.result()
            except Exception as e:
                log.error("%s: %s", name, e)
                failed.append(name)

    n_done = write_index(args.out, models)
    log.info("fetched %d, skipped %d, failed %d; %d of %d present locally",
             fetched, len(chosen) - fetched - len(failed), len(failed), n_done, len(names))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
