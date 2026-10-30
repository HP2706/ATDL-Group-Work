from __future__ import annotations
import typing

from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import torch
from torch.utils.data import Dataset


class Example(BaseModel):
    example_id: str
    object_id: str
    split: str
    frames: list[str] = Field(min_length=1)
    masks: list[str] = Field(min_length=1)
    video: str
    target: str
    ground_truth_mesh: str
    reference_tokens: str
    visibility: str | None = None
    occupancy: str | None = None


def read_manifest(path: Path) -> list[Example]:
    rows = [Example.model_validate_json(line) for line in path.read_text().splitlines() if line.strip()]
    identities: dict[str, str] = {}
    ids: set[str] = set()
    for row in rows:
        if row.example_id in ids:
            raise ValueError("Duplicate example ID")
        ids.add(row.example_id)
        if len(row.frames) != len(row.masks):
            raise ValueError("Each selected frame needs its object mask")
        if row.object_id in identities and identities[row.object_id] != row.split:
            raise ValueError("Object identity leaked across splits")
        identities[row.object_id] = row.split
    return rows


class NaviDataset(Dataset):
    """Existing prepared NAVI manifests through the common sample contract."""

    def __init__(self, cfg: typing.Any, split: str, train: bool, clip_len: int | None = None) -> None:
        self.cfg, self.train = cfg, train
        self.rows = [row for row in read_manifest(Path(cfg.data.prepared) / "manifest.jsonl") if row.split == split]
        if not self.rows:
            raise ValueError(f"No NAVI samples for {split}")
        self.clip_len = clip_len or cfg.data.frames
        self.index = [{"clip_id": row.example_id, "asset_id": row.object_id,
                       "k": 1, "r": 0, "rules": ["reference"]} for row in self.rows]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, position: int | tuple[int, int]) -> dict[str, Any]:
        from video3d.data.samples import preprocess_video, tubelet_offsets
        i, epoch = position if isinstance(position, tuple) else (position, 0)
        row = self.rows[i]
        ids = np.linspace(0, len(row.frames) - 1, self.clip_len).round().astype(int)
        images = [np.array(ImageOps.exif_transpose(Image.open(row.frames[t])).convert("RGB")) for t in ids]
        masks = [np.asarray(Image.open(row.masks[t]).convert("L")) > 0 for t in ids]
        if any(image.shape[:2] != mask.shape for image, mask in zip(images,masks)):
            raise ValueError(f"NAVI RGB/mask mismatch: {row.example_id}")
        rng = np.random.default_rng([self.cfg.seed,epoch,i]) if self.train else None
        frames, frame_masks, _ = preprocess_video(images, masks, self.cfg.data, rng)
        target = torch.load(row.target,map_location="cpu",weights_only=True)["shape"]
        return {"inputs": {"frames":frames,"frame_masks":frame_masks,
                           "dt":torch.from_numpy(tubelet_offsets(self.clip_len,0)),"ref_pos":0,
                           "ref_rgb":torch.from_numpy(images[0].copy()),
                           "ref_mask":torch.from_numpy(masks[0].copy())},
                "target":target,
                "info": {**self.index[i],"frame_ids":ids.tolist(),"reference_rules":["reference"],
                         "visible_frac":None,"amodal_px":None,"arc_span_deg":None,"num_objects":1,
                         "ground_truth_mesh":row.ground_truth_mesh,"visibility":row.visibility,
                         "video":row.video}}

    def target_voxels(self, asset_id: str) -> np.ndarray:
        row=next(row for row in self.rows if row.object_id==asset_id)
        if row.occupancy is None:
            raise ValueError(f"NAVI occupancy missing for {asset_id}")
        return np.load(row.occupancy).astype(bool)
