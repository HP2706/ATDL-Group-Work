"""Frozen video/image encoders and per-token mask coverage.

Token order everywhere is (t, h, w): time-major, then rows, then columns.
VJEPAEncoder and DinoEncoder are added in P5 (they need the GPU environment).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def token_coverage(mask_crop: torch.Tensor, patch: int = 16, tubelet: int = 2) -> torch.Tensor:
    """Fraction of each token covered by the target's visible mask, in (t, h, w) token order.

    mask_crop: (B, K, H, W) fractional visible mask on the clip crop. Average-pools per patch,
    then averages the frames of each tubelet. K = 1 gives one frame of tokens (image tokenizer).
    Returns (B, T * H/patch * W/patch) with T = K / tubelet (or 1 if K = 1).
    V-JEPA: patch 16, tubelet 2. DINOv2: patch 14, tubelet 1 (C1) or 2 (C2 pair averaging).
    """
    B, K, H, W = mask_crop.shape
    assert H % patch == 0 and W % patch == 0, f"crop {H}x{W} not divisible by patch {patch}"
    h, w = H // patch, W // patch
    m = F.avg_pool2d(mask_crop.flatten(0, 1)[:, None].float(), patch).view(B, K, h, w)
    if K > 1 and tubelet > 1:
        assert K % tubelet == 0, f"K={K} not divisible by tubelet {tubelet}"
        m = m.view(B, K // tubelet, tubelet, h, w).mean(2)
    return m.flatten(1)


def per_token(x: torch.Tensor, tokens_per_step: int) -> torch.Tensor:
    """Repeat a per-tubelet value (B, T) for every token of its tubelet -> (B, T * tokens_per_step)."""
    return x.repeat_interleave(tokens_per_step, dim=1)
