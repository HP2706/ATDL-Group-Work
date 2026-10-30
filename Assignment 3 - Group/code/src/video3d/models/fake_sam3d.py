"""FakeSam3D: the Sam3DGeometry interface on a tiny frozen random transformer (CPU, seconds).

Mirrors the real model's structure where it matters for our code: named conditioning sets
concatenated in dict order, shape tokens that attend only to themselves (so v_shape ignores
the layout input), a zero null-condition for CFG, and a shortcut / flow sampler switch.
"""

from __future__ import annotations
import typing

import torch
import torch.nn as nn
import torch.nn.functional as F
from video3d.config import SamplerConfig
from torch.utils.checkpoint import checkpoint


class FakeSam3D(nn.Module):
    """Tiny frozen random stand-in for Sam3DGeometry (tests and dry runs)."""

    QUERY_SETS = ["crop", "full"]

    def __init__(self, d: int = 32, n_layers: int = 2, heads: int = 4, img: int = 32,
                 patch: int = 8, activation_checkpointing: bool = False, seed: int = 0) -> None:
        super().__init__()
        self.img, self.patch = img, patch
        self.latent_shape = (8, 4, 4, 4)        # (C, r, r, r): 8 = 2x2x2 sub-cells of an 8^3 grid
        self.layout_shape = {"layout": (1, 8)}  # one layout token
        self.activation_checkpointing = activation_checkpointing
        with torch.random.fork_rng():           # fixed weights without touching the global RNG
            torch.manual_seed(seed)
            self.patch_embed = nn.Linear(3 * patch * patch, d)      # stands in for DINOv2
            self.mask_embed = nn.Linear(patch * patch, d)
            self.modality = nn.Parameter(torch.randn(3, d) * 0.5)   # crop, full, mask
            self.latent_in, self.latent_out = nn.Linear(8, d), nn.Linear(d, 8)
            self.layout_in, self.layout_out = nn.Linear(8, d), nn.Linear(d, 8)
            self.tau_embed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
            self.pos = nn.Parameter(torch.randn(64, d) * 0.5)
            self.layers = nn.ModuleList(
                nn.TransformerDecoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True,
                                           norm_first=True) for _ in range(n_layers))
        self.requires_grad_(False)
        self.eval()

    # ---- conditioning ----

    def _patch_tokens(self, x: torch.Tensor) -> torch.Tensor:
        """(B, C, img, img) -> (B, (img/patch)^2, C * patch^2) in raster order."""
        return F.unfold(x, self.patch, stride=self.patch).transpose(1, 2)

    @torch.no_grad()
    def condition(self, rgb: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """rgb (B, H, W, 3) uint8, mask (B, H, W) bool -> crop / full / mask token sets."""
        img = rgb.permute(0, 3, 1, 2).float() / 255.0
        full = F.interpolate(img, size=(self.img, self.img), mode="bilinear", align_corners=False)
        crops = []
        for b in range(len(img)):
            ys, xs = torch.nonzero(mask[b], as_tuple=True)
            patch = img[b:b + 1, :, ys.min():ys.max() + 1, xs.min():xs.max() + 1]
            crops.append(F.interpolate(patch, size=(self.img, self.img), mode="bilinear",
                                       align_corners=False))
        m = F.interpolate(mask[:, None].float(), size=(self.img, self.img), mode="area")
        return {"crop": self.patch_embed(self._patch_tokens(torch.cat(crops))) + self.modality[0],
                "full": self.patch_embed(self._patch_tokens(full)) + self.modality[1],
                "mask": self.mask_embed(self._patch_tokens(m)) + self.modality[2]}

    # ---- flow model ----

    def _layer(self, layer: typing.Any, h: typing.Any, c: typing.Any, self_mask: typing.Any) -> typing.Any:
        return layer(h, c, tgt_mask=self_mask)

    def velocity(self, x_tau: typing.Any, tau: typing.Any, cond: dict, layout_in: dict) -> typing.Any:
        """x_tau (B, C, r, r, r), tau (B,), layout_in {"layout": (B, 1, 8)} -> (v_shape, v_layout)."""
        B = x_tau.shape[0]
        t = self.tau_embed(tau.float().view(B, 1, 1))
        lat = self.latent_in(x_tau.flatten(2).transpose(1, 2)) + self.pos + t
        lay = self.layout_in(layout_in["layout"]) + t
        c = torch.cat(list(cond.values()), dim=1)
        h = torch.cat([lat, lay], dim=1)
        n = lat.shape[1]
        self_mask = torch.zeros(h.shape[1], h.shape[1], dtype=torch.bool, device=h.device)
        self_mask[:n, n:] = True                          # shape tokens attend only to shape
        for layer in self.layers:
            if self.activation_checkpointing and torch.is_grad_enabled():
                h = checkpoint(self._layer, layer, h, c, self_mask, use_reentrant=False)
            else:
                h = self._layer(layer, h, c, self_mask)
        v_shape = self.latent_out(h[:, :n]).transpose(1, 2).reshape(x_tau.shape)
        return v_shape, {"layout": self.layout_out(h[:, n:])}

    @torch.no_grad()
    def sample(self, cond: dict, seeds: list[int], sampler_cfg: SamplerConfig) -> typing.Any:
        """Euler ODE from noise (one seed per sample); CFG with a zero null condition in flow mode."""
        device = next(iter(cond.values())).device
        noise = lambda shape, s: torch.randn(shape, generator=torch.Generator().manual_seed(s))
        x = torch.stack([noise(self.latent_shape, s) for s in seeds]).to(device)
        lay = {"layout": torch.stack([noise(self.layout_shape["layout"], s + 1) for s in seeds]).to(device)}
        steps = int(sampler_cfg.steps)
        scale = 0.0 if sampler_cfg.mode == "shortcut" or sampler_cfg.cfg_scale is None\
            else float(sampler_cfg.cfg_scale)
        null = {name: torch.zeros_like(c) for name, c in cond.items()}
        for i in range(steps):
            tau = torch.full((len(seeds),), i / steps, device=device)
            v, l = self.velocity(x, tau, cond, lay)
            if scale > 0:
                v_u, l_u = self.velocity(x, tau, null, lay)
                v, l = (1 + scale) * v - scale * v_u, {"layout": (1 + scale) * l["layout"] - scale * l_u["layout"]}
            x = x + v / steps
            lay = {"layout": lay["layout"] + l["layout"] / steps}
        return x, lay

    @torch.no_grad()
    def zero_shot_layout(self, rgb: torch.Tensor, mask: torch.Tensor) -> dict:
        """Stand-in for SAM 3D's own zero-shot layout: zeros."""
        return {"layout": torch.zeros(len(rgb), *self.layout_shape["layout"], device=rgb.device)}

    # ---- sparse-structure VAE stand-in ----

    @torch.no_grad()
    def encode_voxels(self, vox: torch.Tensor) -> torch.Tensor:
        """(B, 64, 64, 64) bool -> (B, 8, 4, 4, 4): 8^3 occupancy fractions, space-to-depth by 2."""
        occ = F.avg_pool3d(vox.float()[:, None], 8)                       # (B, 1, 8, 8, 8)
        B = occ.shape[0]
        z = occ.view(B, 4, 2, 4, 2, 4, 2).permute(0, 2, 4, 6, 1, 3, 5).reshape(B, 8, 4, 4, 4)
        return 4 * z - 1

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """Inverse of encode_voxels, thresholded and upsampled to 64^3 (crude; fake only)."""
        B = z.shape[0]
        occ = ((z + 1) / 4).view(B, 2, 2, 2, 4, 4, 4).permute(0, 4, 1, 5, 2, 6, 3).reshape(B, 8, 8, 8)
        vox = occ > 0.05
        for dim in (1, 2, 3):
            vox = vox.repeat_interleave(8, dim=dim)
        return vox
