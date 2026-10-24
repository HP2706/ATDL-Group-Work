"""T7 visual review: per-clip mp4 with mask overlays plus a contact sheet.

Usage:
    uv run python -m sam3dvid.review --dataset dataset --split eval --num 20 --out dataset/review
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from .clipio import Clip

PALETTE = np.array([[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48]], np.uint8)


def overlay(clip: Clip, t: int) -> np.ndarray:
    """RGB with visible masks tinted and amodal outlines drawn per object, plus a depth inset."""
    img = clip.rgb(t).astype(np.float32)
    vis = clip.visible(t)
    for k in range(1, clip.num_objects + 1):
        color = PALETTE[(k - 1) % len(PALETTE)].astype(np.float32)
        m = vis == k
        img[m] = 0.6 * img[m] + 0.4 * color
    img = img.astype(np.uint8)
    for k in range(1, clip.num_objects + 1):
        contours, _ = cv2.findContours(clip.amodal(k, t).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img, contours, -1, PALETTE[(k - 1) % len(PALETTE)].tolist(), 1)
    depth = clip.depth(t)
    if np.isfinite(depth).any():
        lo, hi = np.nanmin(depth), np.nanmax(depth)
        d = np.where(np.isfinite(depth), 255 * (1 - (depth - lo) / max(hi - lo, 1e-6)), 0).astype(np.uint8)
        inset = cv2.resize(cv2.applyColorMap(d, cv2.COLORMAP_VIRIDIS)[..., ::-1], (128, 128), interpolation=cv2.INTER_NEAREST)
        img[:128, -128:] = inset
    cv2.putText(img, f"t={t}", (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    return img


def contact_sheet(clip: Clip, frames=(0, 5, 10, 15, 16, 21, 26, 31), scale: float = 0.5) -> np.ndarray:
    tiles = []
    for t in frames:
        rgb = clip.rgb(t)
        ov = overlay(clip, t)
        pair = np.concatenate([rgb, ov], axis=0)
        tiles.append(cv2.resize(pair, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA))
    sheet = np.concatenate(tiles, axis=1)
    m = clip.meta
    title = (
        f"{m['clip_id']}  N={m['num_objects']}  arc={m['arc_span_deg']:.0f}deg  "
        f"el={m['start_elevation_deg']:.0f}+{m['elevation_drift_deg']:.0f}  bg={m['background']['image_id']}"
    )
    header = np.zeros((28, sheet.shape[1], 3), np.uint8)
    cv2.putText(header, title, (6, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    return np.concatenate([header, sheet], axis=0)


def write_review(clip: Clip, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    name = f"{clip.meta['split']}_{clip.meta['clip_id']}"
    frames = [np.concatenate([clip.rgb(t), overlay(clip, t)], axis=1) for t in range(clip.num_frames)]
    imageio.mimwrite(out / f"{name}.mp4", frames, fps=8, quality=8, macro_block_size=1)
    imageio.imwrite(out / f"{name}_sheet.jpg", contact_sheet(clip), quality=90)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", type=Path, default=Path("dataset"))
    ap.add_argument("--split", default="eval")
    ap.add_argument("--num", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path("dataset/review"))
    args = ap.parse_args()
    entries = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines() if line]
    entries = [e for e in entries if e["split"] == args.split]
    random.Random(args.seed).shuffle(entries)
    for e in entries[: args.num]:
        write_review(Clip(args.dataset / e["path"]), args.out)
        print(f"reviewed {e['path']}")


if __name__ == "__main__":
    main()
