"""T6 SAM 3D smoke test (spec Section 7): input export for the GPU run.

Usage:
    uv run python -m sam3dvid.t6 export --split eval --num-clips 20 --out t6/inputs

The export holds, per clip, frame 0's RGB, one visible mask per object and a GT pointmap in
SAM 3D's input convention, so the GPU run (scripts/t6/run_sam3d.py) needs no code from this
package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from .clipio import Clip

# SAM 3D's pointmaps are in PyTorch3D camera axes (x left, y up, z forward); see
# camera_to_pytorch3d_camera in sam3d_objects/pipeline/inference_pipeline_pointmap.py.
PYTORCH3D_FROM_OPENCV = np.diag([-1.0, -1.0, 1.0])


def pointmap_opencv(depth: np.ndarray, K: np.ndarray) -> np.ndarray:
    """(H, W, 3) camera-frame points from z-depth; NaN where depth is NaN."""
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w), np.arange(h))
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    return np.stack([x, y, depth], axis=-1)


def export_inputs(clips: list[Clip], out: Path, t: int = 0) -> None:
    out.mkdir(parents=True, exist_ok=True)
    index = []
    for clip in clips:
        d = out / clip.path.name
        d.mkdir(exist_ok=True)
        Image.fromarray(clip.rgb(t)).save(d / "image.png")
        visible = clip.visible(t)
        objects = []
        for k in range(1, clip.num_objects + 1):
            mask = visible == k
            Image.fromarray(mask.astype(np.uint8) * 255).save(d / f"mask_{k}.png")
            objects.append({"k": k, "visible_px": int(mask.sum())})
        points = pointmap_opencv(clip.depth(t), clip.K) @ PYTORCH3D_FROM_OPENCV.T
        np.savez_compressed(d / "pointmap.npz", pointmap=points.astype(np.float32))
        index.append({"clip_id": clip.path.name, "frame": t, "objects": objects})
    (out / "index.json").write_text(json.dumps(index, indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    ex = sub.add_parser("export")
    ex.add_argument("--dataset", type=Path, default=Path("dataset"))
    ex.add_argument("--split", default="eval")
    ex.add_argument("--num-clips", type=int, default=20)
    ex.add_argument("--out", type=Path, default=Path("t6/inputs"))
    args = ap.parse_args()
    if args.cmd == "export":
        paths = sorted((args.dataset / "clips" / args.split).iterdir())[: args.num_clips]
        export_inputs([Clip(p) for p in paths], args.out)


if __name__ == "__main__":
    main()
