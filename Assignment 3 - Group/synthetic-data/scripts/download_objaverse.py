"""Download a seeded subset of the TRELLIS-500K Objaverse-XL (Sketchfab) training assets.

Usage:
    uv run python scripts/download_objaverse.py [--out data/raw/objaverse] [--limit N]
        [--seed 0] [--min-aesthetic 5.5] [--workers 8]

The candidate list is TRELLIS-500K's ObjaverseXL_sketchfab.csv (HuggingFace dataset
JeffreyXiang/TRELLIS-500K; columns sha256, file_identifier, aesthetic_score, captions), filtered to
aesthetic_score >= --min-aesthetic. The GLB files are Objaverse 1.0 objects from the HuggingFace
dataset allenai/objaverse, located through its object-paths.json.gz; this is the same route as
TRELLIS's datasets/ObjaverseXL.py (objaverse.xl), minus the GitHub subset. A downloaded file whose
sha256 differs from the CSV is discarded, as TRELLIS does.

With --limit, the subset is the first N candidates of a seeded shuffle of the sha256-sorted
candidate list, so a larger limit extends a smaller one. GLBs go to <out>/glbs/<uid>.glb via .part
files (resumed after interruption); existing files are skipped. Licenses and authors come from the
Objaverse 1.0 per-chunk metadata, cached compactly under <out>/meta/. <out>/index.json lists the
selection with sha256, aesthetic score, license and source URL per object; the order of its
"selection" list is the order `python -m sam3dvid.assets --limit N` uses.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import logging
import random
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger("download_objaverse")

HF = "https://huggingface.co/datasets"
CSV_URL = f"{HF}/JeffreyXiang/TRELLIS-500K/resolve/main/ObjaverseXL_sketchfab.csv"
OBJAVERSE = f"{HF}/allenai/objaverse/resolve/main"
CHUNK = 1 << 20


class DownloadError(RuntimeError):
    pass


def fetch(url: str, dest: Path, retries: int = 6, timeout: float = 60.0) -> None:
    """Download url to dest via dest.part, resuming a partial file and retrying with backoff."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            have = part.stat().st_size if part.exists() else 0
            headers = {"Range": f"bytes={have}-"} if have else {}
            with requests.get(url, stream=True, timeout=timeout, headers=headers) as r:
                if r.status_code == 416:  # the .part is already complete
                    r.close()
                    part.replace(dest)
                    return
                if r.status_code == 404:
                    raise DownloadError("HTTP 404")
                if r.status_code >= 500 or r.status_code == 429:
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code not in (200, 206):
                    raise DownloadError(f"HTTP {r.status_code}")
                resumed = r.status_code == 206
                expected = int(r.headers.get("Content-Length", 0)) + (have if resumed else 0)
                with part.open("ab" if resumed else "wb") as f:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
            if expected and part.stat().st_size != expected:
                raise RuntimeError("size mismatch")
            part.replace(dest)
            return
        except DownloadError:
            part.unlink(missing_ok=True)
            raise
        except (requests.RequestException, RuntimeError) as e:
            msg = str(e) if isinstance(e, RuntimeError) else type(e).__name__
            if attempt == retries:
                raise DownloadError(f"download failed after {retries} attempts ({msg})") from None
            wait = min(2**attempt, 60)
            log.warning("%s: attempt %d failed (%s), retrying in %ds", url.rsplit("/", 1)[-1], attempt, msg, wait)
            time.sleep(wait)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(CHUNK), b""):
            h.update(block)
    return h.hexdigest()


def uid_of(file_identifier: str) -> str:
    return file_identifier.rstrip("/").rsplit("/", 1)[-1]


def parse_captions(raw: str) -> list[str]:
    """Captions as a list: a JSON list is parsed, any other non-empty string becomes a one-item list."""
    if not raw:
        return []
    try:
        value = json.loads(raw)
    except json.JSONDecodeError:
        return [raw]
    return [str(c) for c in value] if isinstance(value, list) else [str(value)]


def read_candidates(csv_path: Path, object_paths: dict[str, str], min_aesthetic: float) -> list[dict]:
    """Rows of the TRELLIS CSV with aesthetic_score >= min_aesthetic whose GLB exists in Objaverse 1.0."""
    csv.field_size_limit(1 << 30)  # the captions column can be large
    rows = []
    with csv_path.open(newline="") as f:
        for row in csv.DictReader(f):
            if not row["aesthetic_score"] or float(row["aesthetic_score"]) < min_aesthetic:
                continue
            uid = uid_of(row["file_identifier"])
            if uid not in object_paths:
                continue
            rows.append(
                {
                    "uid": uid,
                    "sha256": row["sha256"],
                    "aesthetic_score": float(row["aesthetic_score"]),
                    "source_url": row["file_identifier"],
                    "captions": parse_captions(row.get("captions", "")),
                }
            )
    return rows


def select(rows: list[dict], limit: int | None, seed: int) -> list[dict]:
    """Return the first limit rows of a seeded shuffle of the sha256-sorted rows."""
    order = sorted(rows, key=lambda r: r["sha256"])
    random.Random(seed).shuffle(order)
    return order if limit is None else order[:limit]


def chunk_metadata(chunk: str, out: Path) -> dict[str, dict]:
    """uid -> license, name, author for one Objaverse 1.0 metadata chunk, cached compactly."""
    cache = out / "meta" / f"{chunk}.json"
    if not cache.exists():
        raw = out / "meta" / f"{chunk}.json.gz"
        fetch(f"{OBJAVERSE}/metadata/{chunk}.json.gz", raw)
        with gzip.open(raw, "rt") as f:
            full = json.load(f)
        raw.unlink()
        compact = {}
        for uid, m in full.items():
            user = m.get("user") or {}
            compact[uid] = {
                "license": m.get("license"),
                "name": m.get("name"),
                "author": user.get("displayName") or user.get("username"),
                "author_url": user.get("profileUrl"),
                "glb_size": ((m.get("archives") or {}).get("glb") or {}).get("size"),
            }
        tmp = cache.with_name(cache.name + ".tmp")
        tmp.write_text(json.dumps(compact))
        tmp.replace(cache)
    return json.loads(cache.read_text())


def download_glb(entry: dict, object_paths: dict[str, str], out: Path) -> bool:
    """Ensure <out>/glbs/<uid>.glb exists with the expected sha256. Returns False if it was already there."""
    dest = out / "glbs" / f"{entry['uid']}.glb"
    if dest.exists():
        return False
    fetch(f"{OBJAVERSE}/{object_paths[entry['uid']]}", dest)
    if sha256_of(dest) != entry["sha256"]:
        dest.unlink()
        raise DownloadError("sha256 differs from the TRELLIS-500K CSV")
    return True


def write_index(out: Path, args: argparse.Namespace, num_candidates: int, entries: list[dict], failed: dict[str, str]) -> int:
    downloaded = [e["uid"] for e in entries if (out / "glbs" / f"{e['uid']}.glb").exists()]
    index = {
        "source": {"csv": CSV_URL, "glbs": f"{OBJAVERSE}/<object-paths.json.gz entry>"},
        "min_aesthetic": args.min_aesthetic,
        "seed": args.seed,
        "num_candidates": num_candidates,
        "num_selected": len(entries),
        "num_downloaded": len(downloaded),
        "licenses": dict(Counter(e["license"] for e in entries)),
        "selection": [e["uid"] for e in entries],
        "downloaded": downloaded,
        "failed": failed,
        "objects": {e["uid"]: e for e in entries},
    }
    tmp = out / "index.json.tmp"
    tmp.write_text(json.dumps(index, indent=1) + "\n")
    tmp.replace(out / "index.json")
    return len(downloaded)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", type=Path, default=Path("data/raw/objaverse"))
    p.add_argument("--limit", type=int, help="number of assets to select, default: all candidates")
    p.add_argument("--seed", type=int, default=0)
    # TRELLIS-500K's own threshold for the Objaverse-XL subsets (dataset_toolkits README table).
    p.add_argument("--min-aesthetic", type=float, default=5.5)
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    out = args.out
    (out / "glbs").mkdir(parents=True, exist_ok=True)
    (out / "meta").mkdir(exist_ok=True)
    csv_path, paths_path = out / "ObjaverseXL_sketchfab.csv", out / "object-paths.json.gz"
    for url, path in [(CSV_URL, csv_path), (f"{OBJAVERSE}/object-paths.json.gz", paths_path)]:
        if not path.exists():
            log.info("fetching %s", path.name)
            fetch(url, path)
    with gzip.open(paths_path, "rt") as f:
        object_paths = json.load(f)

    candidates = read_candidates(csv_path, object_paths, args.min_aesthetic)
    log.info("candidates: %d with aesthetic_score >= %g", len(candidates), args.min_aesthetic)
    chosen = select(candidates, args.limit, args.seed)

    chunks = sorted({object_paths[r["uid"]].split("/")[1] for r in chosen})
    meta: dict[str, dict] = {}
    for chunk in tqdm(chunks, desc="metadata chunks", unit="chunk"):
        meta.update(chunk_metadata(chunk, out))
    entries = []
    for r in chosen:
        m = meta.get(r["uid"], {})
        entries.append({**r, "license": m.get("license"), "name": m.get("name"), "author": m.get("author"),
                        "author_url": m.get("author_url"), "glb_size": m.get("glb_size"),
                        "download_url": f"{OBJAVERSE}/{object_paths[r['uid']]}"})
    log.info("selected %d; licenses: %s", len(entries), dict(Counter(e["license"] for e in entries)))

    failed: dict[str, str] = {}
    write_index(out, args, len(candidates), entries, failed)
    fetched = 0
    with ThreadPoolExecutor(args.workers) as pool:
        futures = {pool.submit(download_glb, e, object_paths, out): e["uid"] for e in entries}
        for fut in tqdm(as_completed(futures), total=len(futures), unit="glb"):
            uid = futures[fut]
            try:
                fetched += fut.result()
            except Exception as e:
                log.error("%s: %s", uid, e)
                failed[uid] = str(e)

    n_done = write_index(out, args, len(candidates), entries, failed)
    log.info("fetched %d, skipped %d, failed %d; %d of %d selected present locally",
             fetched, len(entries) - fetched - len(failed), len(failed), n_done, len(entries))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
