"""Frozen video/image encoders and per-token mask coverage.

Token order everywhere is (t, h, w): time-major, then rows, then columns.
Every encoder exposes `encode(frames (B, K, 3, S, S) in [0, 1]) -> (B, N, d)` plus the attributes
`image_size`, `patch` and `tubelet` (frames averaged per token: 2 for V-JEPA, 1 for DINOv2).
"""

from __future__ import annotations
import typing


import torch
import torch.nn as nn
import torch.nn.functional as F
from video3d.config import ExperimentConfig


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


def kv_tokens(encoder: typing.Any, inputs: dict, kv_source: str) -> typing.Any:
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

    def __init__(self, d: int, image_size: int, patch: int = 16, tubelet: int = 2, seed: int = 0) -> None:
        super().__init__()
        self.image_size, self.patch, self.tubelet = image_size, patch, tubelet
        self.width = d
        with torch.random.fork_rng():
            torch.manual_seed(seed)
            self.proj = nn.Linear(3 * patch * patch, d)
        self.requires_grad_(False)
        self.eval()

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, K, 3, S, S) in [0, 1] -> (B, T * (S/patch)^2, d) in (t, h, w) order."""
        B, K = frames.shape[:2]
        if self.tubelet > 1 and K > 1:
            frames = frames.view(B, K // self.tubelet, self.tubelet, *frames.shape[2:]).mean(2)
        x = F.unfold(frames.flatten(0, 1), self.patch, stride=self.patch).transpose(1, 2)
        return self.proj(x).reshape(B, -1, self.width)


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def imagenet_normalize(frames: torch.Tensor) -> torch.Tensor:
    """(..., 3, H, W) in [0, 1] -> ImageNet-normalised (V-JEPA 2.1 and DINOv2 both use it)."""
    mean = frames.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = frames.new_tensor(IMAGENET_STD).view(3, 1, 1)
    return (frames - mean) / std


class VJEPAEncoder(nn.Module):
    """Frozen V-JEPA 2.1 ViT-L/16 from the pinned facebookresearch/vjepa2 code, run in bf16.

    K = 1 uses the image patch embedding (T = img_temporal_dim_size = 1); K > 1 uses tubelets of 2.
    """

    def __init__(self, cfg: ExperimentConfig, device: torch.device) -> None:
        super().__init__()
        from src.hub import backbones
        backbones.VJEPA_BASE_URL = cfg.encoder.vjepa.weights_url   # pinned code points at localhost
        model, predictor = backbones.__dict__[cfg.encoder.vjepa.name](pretrained=True)
        del predictor
        if cfg.encoder.vjepa.layers == "hierarchical":
            model.out_layers = list(model.hierarchical_layers)     # 4 levels -> 4 x 1024
        else:
            assert cfg.encoder.vjepa.layers == "last", cfg.encoder.vjepa.layers
        assert model.patch_size == 16 and model.tubelet_size == 2, (model.patch_size, model.tubelet_size)
        self.model = model.to(device).eval().requires_grad_(False)
        self.image_size, self.patch, self.tubelet = int(cfg.encoder.vjepa.image_size), 16, 2
        if cfg.encoder.vjepa.video_lora:
            from video3d.adapters import inject_lora
            inject_lora(self.model, r"blocks\.\d+\.attn\.(qkv|proj)", cfg.encoder.vjepa.lora_rank, cfg.encoder.vjepa.lora_alpha)

    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, K, 3, S, S) in [0, 1] -> (B, T * (S/16)^2, 1024 or 4096)."""
        x = imagenet_normalize(frames).permute(0, 2, 1, 3, 4)       # V-JEPA wants (B, 3, T, H, W)
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            out = self.model(x)
        if isinstance(out, list):
            out = torch.cat(out, dim=-1)
        return out.float()


class DinoEncoder(nn.Module):
    """SAM 3D's own frozen DINOv2 at 224 px (16 x 16 patches), one token set per frame."""

    def __init__(self, dino: nn.Module, image_size: int = 224) -> None:
        super().__init__()
        self.model = dino.eval().requires_grad_(False)
        self.patch = int(self.model.patch_embed.patch_size[0])
        self.image_size, self.tubelet = image_size, 1
        assert image_size % self.patch == 0, (image_size, self.patch)

    @torch.no_grad()
    def encode(self, frames: torch.Tensor) -> torch.Tensor:
        """(B, K, 3, S, S) in [0, 1] -> (B, K * (S/14)^2, d) patch tokens (no CLS / registers)."""
        B, K = frames.shape[:2]
        x = imagenet_normalize(frames.flatten(0, 1))
        with torch.autocast(device_type=x.device.type, dtype=torch.bfloat16):
            out = self.model.forward_features(x)["x_norm_patchtokens"]
        return out.float().reshape(B, -1, out.shape[-1])


def build_kv_encoder(cfg: ExperimentConfig, sam3d: nn.Module, device: torch.device) -> nn.Module:
    """The frozen key/value encoder for cfg.encoder.kv_source (fake with cfg.fake_sam3d)."""
    source = cfg.encoder.kv_source
    if cfg.fake_sam3d:
        encoder = FakeEncoder(cfg.adapter.d_vid, cfg.data.crop.size,
                              tubelet=2 if source == "vjepa" else 1).to(device)
        if cfg.encoder.vjepa.video_lora:
            from video3d.adapters import inject_lora
            inject_lora(encoder, "proj", cfg.encoder.vjepa.lora_rank, cfg.encoder.vjepa.lora_alpha)
        return encoder
    if source == "vjepa":
        encoder = VJEPAEncoder(cfg, device)
    elif source in ("dinov2_ref", "dinov2_clip"):
        assert source == "dinov2_ref" or cfg.encoder.dinov2.pair_average, "C2 needs pair_average"
        encoder = DinoEncoder(sam3d.dino_backbone(), cfg.encoder.dinov2.image_size)
    else:
        raise ValueError(f"unknown encoder.kv_source {source!r}")
    width = encoder.encode(torch.zeros(1, 1, 3, encoder.image_size, encoder.image_size,
                                       device=device)).shape[-1]
    assert width == cfg.adapter.d_vid, f"encoder width {width} != adapter.d_vid {cfg.adapter.d_vid}"
    return encoder
