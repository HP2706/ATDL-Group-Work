"""Validated sample identity and persisted evaluation records."""
from __future__ import annotations
from pydantic import BaseModel, ConfigDict


class SampleIdentity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    clip_id: str
    asset_id: str
    k: int
    r: int
    frame_ids: list[int]
    split: str


class EvaluationRecord(BaseModel):
    model_config = ConfigDict(extra="allow")
    schema_version: int = 3
    metric_protocol: str = "surface-voxel-icp-similarity-sum-v1"
    sample_id: str
    label: str
    run: str
    clip_id: str
    asset_id: str
