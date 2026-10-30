"""Build disjoint train/eval background pools from extracted SA-1B images.

Usage:
    uv run python scripts/build_bg_pools.py [--root .] [--src data/raw/sa1b]
        [--out data/bg_pools] [--eval-frac 0.2] [--min-short-side 1024]

--src and --out are resolved against --root, and the "path" field in the
output is relative to --root. Writes train.jsonl, eval.jsonl (sorted by
image_id) and stats.json into --out. Pool membership depends only on the image
ID, so it is stable as archives are added. Images are not copied.
"""

from __future__ import annotations
import typing

import fire
from pydantic import BaseModel
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def eval_fraction_key(image_id: str) -> float:
    """Map an image ID to a stable value in [0, 1)."""
    return int(hashlib.sha1(image_id.encode()).hexdigest()[:16], 16) / 2**64


def scan(root: Path, src: Path, min_short: int, eval_frac: float) -> typing.Any:
    """Return (pools, stats) for all .jpg files under src."""
    pools: dict[str, list[dict]] = {"train": [], "eval": []}
    rejected: Counter[str] = Counter()
    seen: set[str] = set()
    scanned = 0
    for path in sorted(src.rglob("*.jpg")):
        scanned += 1
        image_id = path.stem
        if image_id in seen:
            rejected["duplicate_id"] += 1
            continue
        seen.add(image_id)
        try:
            with Image.open(path) as im:
                w, h = im.size
        except Exception:
            rejected["unreadable"] += 1
            continue
        if w < h:
            rejected["portrait"] += 1
            continue
        if h < min_short:
            rejected["short_side"] += 1
            continue
        pool = "eval" if eval_fraction_key(image_id) < eval_frac else "train"
        pools[pool].append(
            {"image_id": image_id, "path": path.relative_to(root).as_posix(), "width": w, "height": h}
        )
    for recs in pools.values():
        recs.sort(key=lambda r: r["image_id"])
    stats = {
        "src": src.relative_to(root).as_posix(),
        "eval_frac": eval_frac,
        "min_short_side": min_short,
        "scanned": scanned,
        "kept": sum(len(r) for r in pools.values()),
        "rejected": dict(sorted(rejected.items())),
        "pools": {k: len(v) for k, v in pools.items()},
    }
    return pools, stats


class Options(BaseModel):
    root: Path = PROJECT_ROOT
    src: Path = Path('data/raw/sa1b')
    out: Path = Path('data/bg_pools')
    eval_frac: float = 0.2
    min_short_side: int = 1024


def run(root: Path = PROJECT_ROOT, src: Path = Path('data/raw/sa1b'), out: Path = Path('data/bg_pools'), eval_frac: float = 0.2, min_short_side: int = 1024) -> int | None:
    args = Options(**locals())

    if not 0.0 <= args.eval_frac <= 1.0:
        raise ValueError("--eval-frac must be in [0, 1]")
    root = args.root.resolve()
    src = (root / args.src).resolve()
    out = (root / args.out).resolve()
    if not src.is_dir():
        print(f"source dir {src} not found", file=sys.stderr)
        return 1

    pools, stats = scan(root, src, args.min_short_side, args.eval_frac)
    out.mkdir(parents=True, exist_ok=True)
    for name, recs in pools.items():
        with (out / f"{name}.jsonl").open("w") as f:
            for r in recs:
                f.write(json.dumps(r) + "\n")
    (out / "stats.json").write_text(json.dumps(stats, indent=2) + "\n")
    print(json.dumps(stats, indent=2))
    return 0




def main() -> None:
    fire.Fire(run)


if __name__ == "__main__":
    main()
