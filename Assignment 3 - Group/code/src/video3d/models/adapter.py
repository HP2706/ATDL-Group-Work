"""Fusion adapter: gated cross-attention from SAM 3D conditioning tokens to video tokens.

Zero-initialised tanh gates make the adapter an exact identity at step 0 (Flamingo-style).
At init only the gates receive gradient; once they move, all adapter weights train.
"""

from __future__ import annotations
import typing

import torch
import torch.nn as nn
from video3d.config import AdapterConfig


class GatedXAttnBlock(nn.Module):
    """c <- c + tanh(g_attn) * CrossAttn(LN(c), LN(v)); c <- c + tanh(g_ff) * FFN(LN(c))."""

    def __init__(self, d: int = 1024, heads: int = 16, ffn_ratio: float = 2.0) -> None:
        super().__init__()
        self.ln_q, self.ln_kv, self.ln_ff = nn.LayerNorm(d), nn.LayerNorm(d), nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True)
        hid = int(d * ffn_ratio)
        self.ff = nn.Sequential(nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, d))
        self.g_attn = nn.Parameter(torch.zeros(1))  # tanh(0) = 0 -> identity at init
        self.g_ff = nn.Parameter(torch.zeros(1))

    def forward(self, c: typing.Any, v: typing.Any, kv_padding_mask: typing.Any=None,
                video_disabled: bool = False) -> typing.Any:
        kv = self.ln_kv(v)
        a, _ = self.attn(self.ln_q(c), kv, kv, key_padding_mask=kv_padding_mask, need_weights=False)
        if not video_disabled:
            c = c + torch.tanh(self.g_attn) * a
        return c + torch.tanh(self.g_ff) * self.ff(self.ln_ff(c))


class VideoFusionAdapter(nn.Module):
    """Selected conditioning tokens attend to video tokens v = W f + m e_obj + E_time[Δt]."""

    def __init__(self, d_cond: typing.Any=1024, d_vid: typing.Any=1024, n_blocks: typing.Any=2, max_dt: typing.Any=7,
                 use_object_embedding: typing.Any=True, use_time_embedding: typing.Any=True, heads: typing.Any=16, ffn_ratio: typing.Any=2.0) -> None:
        super().__init__()
        self.in_proj = nn.Linear(d_vid, d_cond)
        self.e_obj = nn.Parameter(torch.randn(d_cond) * 0.02) if use_object_embedding else None
        self.e_time = nn.Embedding(2 * max_dt + 1, d_cond) if use_time_embedding else None
        if self.e_time is not None:
            nn.init.normal_(self.e_time.weight, std=0.02)
        self.max_dt = max_dt
        self.blocks = nn.ModuleList(GatedXAttnBlock(d_cond, heads, ffn_ratio) for _ in range(n_blocks))

    def embed_video(self, f_vid: typing.Any, coverage: typing.Any, dt: typing.Any) -> typing.Any:
        """(B, N, d_vid) video features -> (B, N, d_cond) keys/values."""
        v = self.in_proj(f_vid)
        if self.e_obj is not None:
            v = v + coverage[..., None] * self.e_obj
        if self.e_time is not None:
            assert int(dt.abs().max()) <= self.max_dt, f"|Δt| > max_dt={self.max_dt}"
            v = v + self.e_time(dt + self.max_dt)
        return v

    def forward(self, c_sel: typing.Any, f_vid: typing.Any, coverage: typing.Any, dt: typing.Any, kv_padding_mask: typing.Any=None,
                video_disabled: bool = False, diagnostics: dict[str, typing.Any] | None = None) -> typing.Any:
        """c_sel (B, Nc, d_cond); f_vid (B, N, d_vid); coverage (B, N); dt (B, N) long."""
        v = self.embed_video(f_vid, coverage, dt)
        original = c_sel
        for index, blk in enumerate(self.blocks):
            if diagnostics is not None:
                kv = blk.ln_kv(v)
                update, weights = blk.attn(blk.ln_q(c_sel), kv, kv, key_padding_mask=kv_padding_mask,
                                           need_weights=True, average_attn_weights=False)
                gated = torch.tanh(blk.g_attn) * update
                if video_disabled:
                    gated = torch.zeros_like(gated)
                from video3d.diagnostics import attention_statistics
                diagnostics[f"block{index}"] = {
                    "gate": float(torch.tanh(blk.g_attn).item()),
                    "video_update_ratio": float((gated.norm() / original.norm().clamp_min(1e-12)).item()),
                    **attention_statistics(weights, coverage, dt, kv_padding_mask)}
                c_sel = c_sel + gated
                c_sel = c_sel + torch.tanh(blk.g_ff) * blk.ff(blk.ln_ff(c_sel))
            else:
                c_sel = blk(c_sel, v, kv_padding_mask, video_disabled=video_disabled)
        if diagnostics is not None:
            diagnostics["conditioning_change_ratio"] = float(((c_sel-original).norm() / original.norm().clamp_min(1e-12)).item())
        return c_sel

    def gates(self) -> dict[str, float]:
        """tanh of every gate, for logging."""
        return {f"block{i}.{name}": torch.tanh(value).item()
                for i, blk in enumerate(self.blocks) for name, value in (("g_attn", blk.g_attn), ("g_ff", blk.g_ff))}


class NativeConditioning(nn.Identity):
    """Parameter-free checkpoint placeholder for native SAM conditioning."""

    def gates(self) -> dict[str, float]:
        return {}


def build_adapter(cfg: AdapterConfig) -> nn.Module:
    """Construct the adapter from the `adapter` config section."""
    if cfg.strategy == "native":
        return NativeConditioning()
    if cfg.strategy == "pooler":
        from video3d.models.pooler import MaskPooler
        return MaskPooler(cfg)
    return VideoFusionAdapter(d_cond=cfg.d_cond, d_vid=cfg.d_vid, n_blocks=cfg.n_blocks,
                              max_dt=cfg.max_dt, use_object_embedding=cfg.use_object_embedding,
                              use_time_embedding=cfg.use_time_embedding, heads=cfg.heads,
                              ffn_ratio=cfg.ffn_ratio)


def count_parameters(module: nn.Module) -> int:
    """Number of trainable parameters."""
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def pad_video_batch(tokens: list[torch.Tensor], coverage: list[torch.Tensor],
                    dt: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    """Pad per-sample (N_i, D) tokens, (N_i,) coverage and Δt to the batch maximum.

    Returns tokens (B, N, D), coverage (B, N), dt (B, N) and kv_padding_mask (B, N) with
    True = padding (ignored by attention), so a batch can mix clip lengths K.
    """
    n = max(len(x) for x in tokens)
    B, D = len(tokens), tokens[0].shape[1]
    out = {"tokens": tokens[0].new_zeros(B, n, D), "coverage": coverage[0].new_zeros(B, n),
           "dt": dt[0].new_zeros(B, n), "kv_padding_mask": torch.ones(B, n, dtype=torch.bool,
                                                                      device=tokens[0].device)}
    for i, (x, m, d) in enumerate(zip(tokens, coverage, dt)):
        out["tokens"][i, :len(x)], out["coverage"][i, :len(x)], out["dt"][i, :len(x)] = x, m, d
        out["kv_padding_mask"][i, :len(x)] = False
    return out


@torch.no_grad()
def attention_mass_on_object(adapter: VideoFusionAdapter, c_sel: typing.Any, f_vid: typing.Any, coverage: typing.Any, dt: typing.Any,
                             kv_padding_mask: typing.Any=None, threshold: float = 0.5) -> list[float]:
    """Per block: mean attention mass (over heads and queries) on tokens with coverage > threshold."""
    v = adapter.embed_video(f_vid, coverage, dt)
    on_object = (coverage > threshold).float()[:, None, :]          # (B, 1, N)
    masses = []
    for blk in adapter.blocks:
        kv = blk.ln_kv(v)
        _, w = blk.attn(blk.ln_q(c_sel), kv, kv, key_padding_mask=kv_padding_mask,
                        need_weights=True)                          # (B, Nc, N), head-averaged
        masses.append((w * on_object).sum(-1).mean().item())
        c_sel = blk(c_sel, v, kv_padding_mask)
    return masses
