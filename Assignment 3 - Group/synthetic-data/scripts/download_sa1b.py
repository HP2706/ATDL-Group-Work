"""Download SA-1B tar archives and extract only their JPEG images.

Usage:
    uv run python scripts/download_sa1b.py [--links data/sa1b_links.txt]
        [--out data/raw/sa1b] [--archives sa_000000.tar ...] [--keep-tar]

Each archive is extracted into <out>/<archive_stem>/ and marked complete with a
.done file; archives that already have the marker are skipped. URLs from the
links file are signed and are never logged.
"""

from __future__ import annotations

import argparse
import logging
import shutil
import sys
import tarfile
import time
from pathlib import Path

import requests
from tqdm import tqdm

log = logging.getLogger("download_sa1b")

CHUNK = 1 << 20
DONE_MARKER = ".done"


def read_links(path: Path) -> dict[str, str]:
    """Return an ordered mapping of archive name to URL from the SA-1B links file."""
    links: dict[str, str] = {}
    with path.open() as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if i == 0 and parts[0] == "file_name":
                continue
            if len(parts) != 2:
                raise ValueError(f"{path}:{i + 1}: expected 2 tab-separated fields")
            links[parts[0]] = parts[1]
    return links


def download(url: str, dest: Path, name: str, retries: int = 8, timeout: float = 60.0) -> None:
    """Stream url to dest, resuming from dest.part across retries. Logs only name."""
    part = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        have = part.stat().st_size if part.exists() else 0
        headers = {"Range": f"bytes={have}-"} if have else {}
        try:
            with requests.get(url, headers=headers, stream=True, timeout=timeout) as r:
                if r.status_code == 416 and have:
                    break
                if r.status_code not in (200, 206):
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code == 200 and have:
                    log.info("%s: server ignored resume, restarting", name)
                    have = 0
                total = int(r.headers.get("Content-Length", 0)) + have or None
                with part.open("ab" if have else "wb") as f, tqdm(
                    total=total, initial=have, unit="B", unit_scale=True, desc=name
                ) as bar:
                    for chunk in r.iter_content(CHUNK):
                        f.write(chunk)
                        bar.update(len(chunk))
                if total is not None and part.stat().st_size < total:
                    raise RuntimeError("connection closed early")
            break
        # requests exceptions embed the signed URL in their message, so log only the type.
        except (requests.RequestException, RuntimeError) as e:
            msg = str(e) if isinstance(e, RuntimeError) else type(e).__name__
            if attempt == retries:
                raise RuntimeError(f"{name}: download failed after {retries} attempts ({msg})") from None
            wait = min(2**attempt, 60)
            log.warning("%s: attempt %d failed (%s), retrying in %ds", name, attempt, msg, wait)
            time.sleep(wait)
    part.replace(dest)


def extract_jpgs(tar_path: Path, dest_dir: Path) -> int:
    """Extract every .jpg in tar_path flat into dest_dir and write the .done marker.

    Returns the number of images extracted. Member paths are reduced to their
    basename so the archive cannot write outside dest_dir.
    """
    dest_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    with tarfile.open(tar_path, "r|*") as tar:
        for m in tar:
            name = Path(m.name).name
            if not m.isfile() or not name.lower().endswith(".jpg") or name.startswith("."):
                continue
            src = tar.extractfile(m)
            tmp = dest_dir / (name + ".tmp")
            with tmp.open("wb") as out:
                shutil.copyfileobj(src, out, CHUNK)
            tmp.replace(dest_dir / name)
            n += 1
    (dest_dir / DONE_MARKER).write_text(f"{n}\n")
    return n


def process_archive(name: str, url: str | None, out: Path, keep_tar: bool) -> None:
    """Download (unless the tar is already present) and extract one archive."""
    stem = Path(name).stem
    dest_dir = out / stem
    if (dest_dir / DONE_MARKER).exists():
        log.info("%s: already extracted, skipping", name)
        return
    tar_path = out / name
    if not tar_path.exists():
        if url is None:
            raise KeyError(f"{name}: not found in links file")
        log.info("%s: downloading", name)
        download(url, tar_path, name)
    log.info("%s: extracting jpgs to %s", name, dest_dir)
    n = extract_jpgs(tar_path, dest_dir)
    log.info("%s: extracted %d images", name, n)
    if not keep_tar:
        tar_path.unlink()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--links", type=Path, default=Path("data/sa1b_links.txt"))
    p.add_argument("--out", type=Path, default=Path("data/raw/sa1b"))
    p.add_argument("--archives", nargs="+", help="archive names, default: first entry in the links file")
    p.add_argument("--keep-tar", action="store_true")
    args = p.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    if not args.links.exists():
        log.error("links file %s not found", args.links)
        return 1
    links = read_links(args.links)
    if not links:
        log.error("links file %s has no entries", args.links)
        return 1
    archives = args.archives or [next(iter(links))]
    missing = [a for a in archives if a not in links]
    if missing:
        log.error("not in links file: %s", ", ".join(missing))
        return 1

    args.out.mkdir(parents=True, exist_ok=True)
    for name in archives:
        process_archive(name, links[name], args.out, args.keep_tar)
    return 0


if __name__ == "__main__":
    sys.exit(main())
