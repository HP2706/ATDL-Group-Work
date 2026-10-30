"""Reading and writing the per-clip folder format (spec Section 5)."""

from __future__ import annotations

import json
from functools import cached_property
from pathlib import Path

import numpy as np
from PIL import Image


def png(path: Path, array: np.ndarray) -> None:
    Image.fromarray(array).save(path, compress_level=6)


def write_clip(
    out: Path,
    rgb: np.ndarray,
    rgba_fg: np.ndarray,
    visible: np.ndarray,
    amodal: np.ndarray,
    depth: np.ndarray,
    cameras: dict,
    objects: list[dict],
    meta: dict,
) -> None:
    """Write arrays shaped (T,H,W,3), (T,H,W,4), (T,H,W), (N,T,H,W) bool, (T,H,W) float32."""
    for sub in ("rgb", "rgba_fg", "mask_visible", "depth"):
        (out / sub).mkdir(parents=True)
    for t in range(rgb.shape[0]):
        name = f"{t:03d}"
        png(out / "rgb" / f"{name}.png", rgb[t])
        png(out / "rgba_fg" / f"{name}.png", rgba_fg[t])
        png(out / "mask_visible" / f"{name}.png", visible[t])
        np.savez_compressed(out / "depth" / f"{name}.npz", depth=depth[t].astype(np.float32))
    for k in range(amodal.shape[0]):
        d = out / "mask_amodal" / f"obj{k + 1}"
        d.mkdir(parents=True)
        for t in range(amodal.shape[1]):
            png(d / f"{t:03d}.png", amodal[k, t].astype(np.uint8) * 255)
    for name, payload in (("cameras", cameras), ("objects", objects), ("meta", meta)):
        (out / f"{name}.json").write_text(json.dumps(payload, indent=1))


class Clip:
    """Lazy reader for one clip folder; frames are 0-based, objects k are 1-based."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    @cached_property
    def cameras(self) -> dict:
        return json.loads((self.path / "cameras.json").read_text())

    @cached_property
    def objects(self) -> list[dict]:
        objs = json.loads((self.path / "objects.json").read_text())
        for o in objs:
            for key in ("R_wo", "t_wo", "s_wo"):
                o[key] = np.array(o[key])
        return objs

    @cached_property
    def meta(self) -> dict:
        return json.loads((self.path / "meta.json").read_text())

    @property
    def K(self) -> np.ndarray:
        return np.array(self.cameras["K"])

    @cached_property
    def T_cw(self) -> np.ndarray:
        return np.array([f["T_cw"] for f in self.cameras["frames"]])

    @property
    def num_frames(self) -> int:
        return len(self.cameras["frames"])

    @property
    def num_objects(self) -> int:
        return len(self.objects)

    @property
    def dataset_root(self) -> Path:
        """clips/{split}/{clip_id} sits three levels below the dataset root."""
        return self.path.parents[2]

    def asset_path(self, k: int, key: str) -> Path:
        return self.dataset_root / self.objects[k - 1][key]

    def rgb(self, t: int) -> np.ndarray:
        return np.asarray(Image.open(self.path / "rgb" / f"{t:03d}.png"))

    def rgba_fg(self, t: int) -> np.ndarray:
        return np.asarray(Image.open(self.path / "rgba_fg" / f"{t:03d}.png"))

    def visible(self, t: int) -> np.ndarray:
        return np.asarray(Image.open(self.path / "mask_visible" / f"{t:03d}.png"))

    def amodal(self, k: int, t: int) -> np.ndarray:
        return np.asarray(Image.open(self.path / "mask_amodal" / f"obj{k}" / f"{t:03d}.png")) > 0

    def depth(self, t: int) -> np.ndarray:
        with np.load(self.path / "depth" / f"{t:03d}.npz") as f:
            return f["depth"]


def preserve_existing(path: Path) -> None:
    """Move generated output aside before an explicit replacement or retry."""
    from uuid import uuid4
    if path.exists():
        path.rename(path.with_name(path.name + ".preserved-" + uuid4().hex[:12]))
