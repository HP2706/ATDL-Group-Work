"""Append aligned video and mask summaries while preserving native conditioning."""
from __future__ import annotations

from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from video3d.config import AdapterConfig
from video3d.models.conditioning import ConditioningSets, VideoFeatures


class VideoMaskAppender(nn.Module):
    def __init__(self, cfg: AdapterConfig) -> None:
        super().__init__()
        width = cfg.d_cond
        if cfg.append_queries < 1 or width % cfg.heads:
            raise ValueError("Positive query count and width divisible by heads required")
        self.heads = cfg.heads
        self.queries = nn.Parameter(torch.randn(cfg.append_queries, width) * .02)
        self.video_projection = nn.Linear(cfg.d_vid, width)
        self.mask_projection = nn.Sequential(nn.Linear(32, width), nn.GELU(), nn.Linear(width, width))
        self.position_projection = nn.Sequential(nn.Linear(3, width), nn.SiLU(), nn.Linear(width, width))
        self.key_norm = nn.LayerNorm(width)
        self.query_norm = nn.LayerNorm(width)
        self.key_projection = nn.Linear(width, width)
        self.video_value = nn.Linear(width, width)
        self.mask_value = nn.Linear(width, width)
        self.video_output = nn.Linear(width, width)
        self.mask_output = nn.Linear(width, width)
        self.video_norm = nn.LayerNorm(width)
        self.mask_norm = nn.LayerNorm(width)
        self.modality = nn.Parameter(torch.randn(2, width) * .02)

    def split_heads(self, value: torch.Tensor) -> torch.Tensor:
        batch, count, width = value.shape
        return value.reshape(batch, count, self.heads, width // self.heads).transpose(1, 2)

    def condition(self, native: ConditioningSets, video: VideoFeatures,
                  query_sets: list[str], video_disabled: bool = False,
                  diagnostics: dict[str, Any] | None = None) -> ConditioningSets:
        if video_disabled:
            return native
        if "video_rgb" in native or "video_mask" in native:
            raise ValueError("Appended conditioning names already exist")
        if video.mask_patches is None or video.positions is None:
            raise ValueError("Video/mask appending requires aligned mask patches and positions")
        batch, count, _ = video.tokens.shape
        if video.mask_patches.shape != (batch, count, 32) or video.positions.shape != (batch, count, 3):
            raise ValueError("Mask patches and positions must align with video tokens")
        valid = torch.ones_like(video.coverage, dtype=torch.bool)
        if video.padding is not None:
            valid = ~video.padding
        if not valid.any(dim=1).all():
            raise ValueError("Cannot pool a sample with no valid video tokens")
        # Invisible frames remain valid evidence. Coverage guides attention through
        # mask features; hard foreground filtering would erase silhouette context.
        position = self.position_projection(video.positions)
        rgb = self.video_projection(video.tokens)
        mask = self.mask_projection(video.mask_patches)
        keys = self.split_heads(self.key_projection(self.key_norm(rgb + mask + position)))
        queries = self.queries[None].expand(batch, -1, -1)
        query_heads = self.split_heads(self.query_norm(queries))
        allowed = valid[:, None, None, :]
        # Identical Q/K, mask and zero dropout give exactly the same attention
        # distribution for both value streams without materializing Q x N weights.
        pooled_rgb = F.scaled_dot_product_attention(
            query_heads, keys, self.split_heads(self.video_value(rgb + position)), attn_mask=allowed)
        pooled_mask = F.scaled_dot_product_attention(
            query_heads, keys, self.split_heads(self.mask_value(mask + position)), attn_mask=allowed)
        width = queries.shape[-1]
        rgb_summary = self.video_norm(queries + self.video_output(
            pooled_rgb.transpose(1, 2).reshape(batch, -1, width))) + self.modality[0]
        mask_summary = self.mask_norm(queries + self.mask_output(
            pooled_mask.transpose(1, 2).reshape(batch, -1, width))) + self.modality[1]
        dtype = next(iter(native.values())).dtype
        if diagnostics is not None:
            diagnostics.update(attention_available=False, appended_tokens=2 * queries.shape[1],
                               reason="Shared-weight video/mask resampling")
        return {**native, "video_rgb": rgb_summary.to(dtype), "video_mask": mask_summary.to(dtype)}

    def gates(self) -> dict[str, float]:
        return {}
