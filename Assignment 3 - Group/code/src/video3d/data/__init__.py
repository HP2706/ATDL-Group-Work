"""Shared dataset contract; generation is part of this same project."""
from __future__ import annotations
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from video3d.config import ExperimentConfig


def build_dataset(cfg: ExperimentConfig, split: str, train: bool,
                  clip_len: int | None = None) -> Any:
    if cfg.data.source == "synthetic":
        from video3d.data.samples import ClipDataset
        return ClipDataset(cfg, split, train, clip_len)
    if cfg.data.source == "navi":
        from video3d.data.navi import NaviDataset
        return NaviDataset(cfg, split, train, clip_len)
    if cfg.data.source in {"gso", "omniobject3d"}:
        from video3d.data.scanned import ScannedDataset
        return ScannedDataset(cfg, split, train, clip_len)
    raise ValueError(f"Unknown data.source: {cfg.data.source}")


def ground_truth(dataset: Any, asset_id: str, cfg: ExperimentConfig) -> Any:
    if cfg.data.source in {"navi", "gso", "omniobject3d"}:
        return dataset.target_voxels(asset_id)
    from video3d.data.targets import target_voxels
    return target_voxels(dataset.voxels_file, asset_id, cfg)
