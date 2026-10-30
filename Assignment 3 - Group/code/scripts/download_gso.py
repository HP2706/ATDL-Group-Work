"""Download Google Scanned Objects models from Gazebo Fuel and extract them.

Usage:
    uv run python scripts/download_gso.py [--out data/raw/gso] [--limit N]
        [--seed 0] [--workers 4]

With --limit, the subset is the first N names after sorting the catalogue and
shuffling it with --seed, so the same seed always yields the same models and a
larger limit extends a smaller one. Each model is extracted into <out>/<name>/
and marked complete with a .done file; complete models are skipped. Source ZIPs
are retained. <out>/index.json records the fixed official evaluation membership.
"""

from __future__ import annotations

import json
import logging
import fire
import random
import shutil
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import quote

import requests

log = logging.getLogger("download_gso")

API = "https://fuel.gazebosim.org/1.0/GoogleResearch/models"
COLLECTION = "https://fuel.gazebosim.org/1.0/GoogleResearch/collections/Scanned%20Objects%20by%20Google%20Research/models"
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
    """Return unique objects in the official collection, not all owner uploads."""
    models: dict[str, dict] = {}
    page = 1
    while True:
        r = get_with_retry(f"{COLLECTION}?per_page={per_page}&page={page}")
        # Fuel answers 404 for a page past the end instead of an empty list.
        if r.status_code == 404:
            break
        r.raise_for_status()
        batch = r.json()
        if not batch:
            break
        models.update({model['name']: model for model in batch})
        page += 1
    snapshot = json.loads((Path(__file__).resolve().parents[1] / 'src/video3d/data/gso-evaluation-objects.json').read_text())
    if set(models) != set(snapshot['objects']):
        raise ValueError('Official GSO collection differs from the fixed evaluation membership; inspect before changing it')
    return [models[name] for name in sorted(models)]


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
    """Validate paths and CRCs, preserve existing extracted files and source archives."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    root = dest_dir.resolve()
    with zipfile.ZipFile(zip_path) as archive:
        for member in archive.infolist():
            if not (dest_dir / member.filename).resolve().is_relative_to(root):
                raise ValueError(f"Unsafe archive member: {member.filename}")
        bad = archive.testzip()
        if bad is not None:
            raise ValueError(f"Corrupt archive member: {bad}")
        for member in archive.infolist():
            destination = dest_dir / member.filename
            if destination.exists():
                if not member.is_dir() and destination.stat().st_size != member.file_size:
                    raise ValueError(f"Existing file differs: {destination}")
                continue
            if member.is_dir():
                destination.mkdir(parents=True, exist_ok=True)
                continue
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = destination.with_name(destination.name + '.extracting')
            with archive.open(member) as source, temporary.open('wb') as output:
                shutil.copyfileobj(source, output, length=1 << 20)
            temporary.replace(destination)
    (dest_dir / DONE_MARKER).write_text("archive CRC verified\n")


def process_model(name: str, out: Path, keep_zip: bool) -> bool:
    """Ensure model name is extracted under out. Returns False if it was already complete."""
    dest_dir = out / name
    if (dest_dir / DONE_MARKER).exists():
        return False
    zip_path = out / f"{name}.zip"
    if not zip_path.exists():
        download(name, zip_path)
    extract(zip_path, dest_dir)
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


def main(out: str = "data/raw/gso", limit: int | None = None, seed: int = 0, workers: int = 4) -> None:
    """Resume the complete GSO catalogue, preserving every source archive."""
    root = Path(out)
    root.mkdir(parents=True, exist_ok=True)
    models = list_models()
    if not models:
        raise RuntimeError("Empty GSO catalogue")
    write_index(root, models)
    chosen = select([m['name'] for m in models], limit, seed)
    failures: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(process_model, name, root, True): name for name in chosen}
        for future in as_completed(futures):
            name = futures[future]
            error = future.exception()
            if error is not None:
                failures[name] = str(error)
            else:
                future.result()
            print(json.dumps({'object': name, 'failed': name in failures}), flush=True)
    completed = write_index(root, models)
    (root / 'download_failures.json').write_text(json.dumps(failures, indent=2))
    print(json.dumps({'catalogue': len(models), 'complete': completed, 'failed': len(failures)}), flush=True)
    if failures:
        raise RuntimeError(f"{len(failures)} GSO downloads failed; see download_failures.json")


if __name__ == '__main__':
    fire.Fire(main)
