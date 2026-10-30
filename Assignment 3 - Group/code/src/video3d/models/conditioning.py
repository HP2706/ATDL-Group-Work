"""Shared contracts for encoders and conditioning strategies."""
from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict
import torch

ConditioningSets = dict[str, torch.Tensor]


class VideoFeatures(BaseModel):
    """Aligned batched tokens; True padding entries never contribute to pooling.

    mask_patches contains a 4x4 mask descriptor per frame, with two temporal
    slices per token. Image encoders repeat their single slice.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")
    tokens: torch.Tensor
    coverage: torch.Tensor
    dt: torch.Tensor
    padding: torch.Tensor | None = None
    mask_patches: torch.Tensor | None = None
    positions: torch.Tensor | None = None


class ConditioningAdapter(Protocol):
    def condition(
        self, native: ConditioningSets, video: VideoFeatures,
        query_sets: list[str], video_disabled: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> ConditioningSets: ...


def rgb_sets(native: ConditioningSets, requested: list[str]) -> list[str]:
    names = requested or [name for name in native
                         if ("rgb" in name or "image" in name)
                         and "mask" not in name and "pointmap" not in name]
    if not names:
        raise ValueError(f"Set adapter.query_sets to RGB conditioning names: {list(native)}")
    missing = [name for name in names if name not in native]
    assert not missing, f"query sets {missing} not in conditioning sets {list(native)}"
    return names
