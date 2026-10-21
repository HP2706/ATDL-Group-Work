from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import numpy as np
from PIL import Image, ImageOps
from pydantic import BaseModel, Field
import torch
from torch.utils.data import Dataset
from .config import DataConfig


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


def clip_tensor(example: Example, config: DataConfig, repeated_reference: bool = False) -> torch.Tensor:
    indices = np.linspace(0, len(example.frames) - 1, config.frames).round().astype(int)
    if repeated_reference:
        indices[:] = indices[0]
    images: list[torch.Tensor] = []
    for index in indices:
        image = ImageOps.exif_transpose(Image.open(example.frames[index])).convert("RGB")
        mask = Image.open(example.masks[index]).convert("L")
        mask_array = np.asarray(mask) > 0
        y, x = np.nonzero(mask_array)
        if not len(x):
            raise ValueError(f"Empty object mask: {example.example_id}")
        box = (int(x.min()), int(y.min()), int(x.max()) + 1, int(y.max()) + 1)
        masked = Image.composite(image, Image.new("RGB", image.size, (127, 127, 127)), mask)
        crop = masked.crop(box)
        side = max(crop.size)
        square = Image.new("RGB", (side, side), (127, 127, 127))
        square.paste(crop, ((side - crop.width) // 2, (side - crop.height) // 2))
        array = np.array(square.resize((config.image_size, config.image_size), Image.Resampling.BICUBIC), copy=True)
        images.append(torch.from_numpy(array).permute(2, 0, 1).float() / 255)
    video = torch.stack(images, dim=1)
    mean = torch.tensor([0.485, 0.456, 0.406])[:, None, None, None]
    std = torch.tensor([0.229, 0.224, 0.225])[:, None, None, None]
    return (video - mean) / std


class Clips(Dataset[dict[str, Any]]):
    def __init__(self, rows: list[Example], config: DataConfig) -> None:
        self.rows, self.config = rows, config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        target = torch.load(row.target, map_location="cpu", weights_only=True)
        reference = torch.load(row.reference_tokens, map_location="cpu", weights_only=True)
        return {"video": clip_tensor(row, self.config), "target": target, "reference": reference}
