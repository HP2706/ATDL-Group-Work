"""Frozen video/image encoders and per-token mask coverage.

Token order everywhere is (t, h, w): time-major, then rows, then columns.
VJEPAEncoder and DinoEncoder are added in P5 (they need the GPU environment).
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig


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


def kv_tokens(encoder, inputs: dict, kv_source: str):
    """One sample's keys/values -> (tokens (N, d), coverage (N,), Δt per token (N,)).

    kv_source: vjepa (tubelets of the clip), dinov2_ref (reference frame only, Δt = 0),
    dinov2_clip (per-frame tokens averaged in pairs, same count as V-JEPA).
    """
    frames, masks, dt = inputs["frames"], inputs["frame_masks"], inputs["dt"]
    if kv_source == "dinov2_ref":
        p = inputs["ref_pos"]
        frames, masks, dt = frames[p:p + 1], masks[p:p + 1], dt.new_zeros(1)
    size = encoder.image_size
    if frames.shape[-1] != size:
        frames = F.interpolate(frames, size=(size, size), mode="bilinear", align_corners=False,
                               antialias=True)
        masks = F.interpolate(masks[:, None], size=(size, size), mode="area")[:, 0]
    tokens = encoder.encode(frames[None])[0]
    K, hw = frames.shape[0], (size // encoder.patch) ** 2
    if kv_source == "dinov2_clip" and K > 1:
        tokens = tokens.view(K // 2, 2, hw, -1).mean(1).flatten(0, 1)
    tubelet = 1 if kv_source == "dinov2_ref" else 2
    coverage = token_coverage(masks[None], encoder.patch, tubelet)[0]
    dt_tok = per_token(dt[None], hw)[0]
    assert len(tokens) == len(coverage) == len(dt_tok), (tokens.shape, coverage.shape, dt_tok.shape)
    return tokens, coverage, dt_tok


class FakeEncoder(nn.Module):
    """Frozen random patch embedding with the encoder interface (tests and dry runs).

    tubelet = 2 mimics V-JEPA (frame pairs averaged, K = 1 -> one frame); 1 mimics DINOv2.
    """

    def __init__(self, d: int, image_size: int, patch: int = 16, tubelet: int = 2, seed: int = 0):
        super().__init__()
        self.image_size, self.patch, self.tubelet = image_size, patch, tubelet
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.proj = nn.Linear(3 * patch * patch, d)
        self.requires_grad_(False)
        self.eval()

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, K, 3, S, S) in [0, 1] -> (B, T * (S/patch)^2, d) in (t, h, w) order."""
        B, K = frames.shape[:2]
        if self.tubelet > 1 and K > 1:
            frames = frames.view(B, K // self.tubelet, self.tubelet, *frames.shape[2:]).mean(2)
        x = F.unfold(frames.flatten(0, 1), self.patch, stride=self.patch).transpose(1, 2)
        return self.proj(x).reshape(B, -1, self.proj.out_features)


def build_kv_encoder(cfg: DictConfig, sam3d: nn.Module, device: torch.device) -> nn.Module:
    """The frozen key/value encoder for cfg.encoder.kv_source (fake with cfg.fake_sam3d)."""
    tubelet = 2 if cfg.encoder.kv_source == "vjepa" else 1
    if cfg.fake_sam3d:
        return FakeEncoder(cfg.adapter.d_vid, cfg.data.crop.size, tubelet=tubelet).to(device)
    raise NotImplementedError("VJEPAEncoder / DinoEncoder arrive in P5")
