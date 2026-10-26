"""SAM 3D Geometry wrapper (Stage 1 only) and FakeSam3D with the same interface.

Interface (both classes):
  condition(rgb, mask)                 -> dict[name -> (B, N, d)] conditioning tokens (no grad)
  velocity(x_tau, tau, cond, layout)   -> (v_shape, v_layout)
  sample(cond, seeds, sampler_cfg)     -> (shape latent, layout); CFG null branch never adapted
  encode_voxels(vox) / decode_latent(z) 64^3 bool <-> normalised shape latent
  zero_shot_layout(rgb, mask)          -> layout used as x1 for the layout tokens in Phase A
  latent_shape, layout_shape           per-sample tensor shapes
`apply_adapter` (a plain function) replaces the query token sets with the adapter output.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import DictConfig
from torch.utils.checkpoint import checkpoint


def apply_adapter(cond: dict, adapter: nn.Module, tokens, coverage, dt, query_sets,
                  kv_padding_mask=None) -> dict:
    """Return a new cond dict where only `query_sets` are replaced by the adapter output."""
    missing = [name for name in query_sets if name not in cond]
    assert not missing, f"query sets {missing} not in conditioning sets {list(cond)}"
    sizes = [cond[name].shape[1] for name in query_sets]
    c_sel = torch.cat([cond[name] for name in query_sets], dim=1)
    out = adapter(c_sel, tokens, coverage, dt, kv_padding_mask)
    return {**cond, **dict(zip(query_sets, out.split(sizes, dim=1)))}


class Sam3DGeometry(nn.Module):
    """Frozen SAM 3D Objects Geometry model. Implemented in P5 after `inspect-sam3d`."""

    def __init__(self, cfg: DictConfig):
        super().__init__()
        raise NotImplementedError("Sam3DGeometry arrives in P5 (see docs/sam3d_internals.md); "
                                  "use --fake-sam3d for dry runs")


class FakeSam3D(nn.Module):
    """Tiny frozen random transformer with the Sam3DGeometry interface (CPU, for tests)."""

    QUERY_SETS = ["crop", "full"]

    def __init__(self, d: int = 32, n_layers: int = 2, heads: int = 4, img: int = 32,
                 patch: int = 8, activation_checkpointing: bool = False, seed: int = 0):
        super().__init__()
        self.img, self.patch = img, patch
        self.latent_shape = (8, 4, 4, 4)    # (C, r, r, r): 8 = 2x2x2 sub-cells of an 8^3 grid
        self.layout_shape = (1, 8)          # one layout token
        self.activation_checkpointing = activation_checkpointing
        with torch.random.fork_rng():       # fixed weights without touching the global RNG
            torch.manual_seed(seed)
            self.patch_embed = nn.Linear(3 * patch * patch, d)      # stands in for DINOv2
            self.mask_embed = nn.Linear(patch * patch, d)
            self.modality = nn.Parameter(torch.randn(3, d) * 0.5)   # crop, full, mask
            self.latent_in, self.latent_out = nn.Linear(8, d), nn.Linear(d, 8)
            self.layout_in, self.layout_out = nn.Linear(8, d), nn.Linear(d, 8)
            self.tau_embed = nn.Sequential(nn.Linear(1, d), nn.SiLU(), nn.Linear(d, d))
            self.pos = nn.Parameter(torch.randn(64, d) * 0.5)
            self.layers = nn.ModuleList(
                nn.TransformerEncoderLayer(d, heads, 2 * d, dropout=0.0, batch_first=True,
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

    def null_condition(self, cond: dict) -> dict:
        """Unconditional branch for CFG (zeros); built from the unadapted structure only."""
        return {name: torch.zeros_like(c) for name, c in cond.items()}

    # ---- flow model ----

    def velocity(self, x_tau, tau, cond, layout_in):
        """x_tau (B, C, r, r, r), tau (B,), layout_in (B, 1, 8) -> (v_shape, v_layout)."""
        B = x_tau.shape[0]
        t = self.tau_embed(tau.float().view(B, 1, 1))
        lat = self.latent_in(x_tau.flatten(2).transpose(1, 2)) + self.pos + t
        lay = self.layout_in(layout_in) + t
        c = torch.cat([cond[name] for name in sorted(cond)], dim=1)
        h = torch.cat([c, lay, lat], dim=1)
        for layer in self.layers:
            if self.activation_checkpointing and torch.is_grad_enabled():
                h = checkpoint(layer, h, use_reentrant=False)
            else:
                h = layer(h)
        n_lat = lat.shape[1]
        v_shape = self.latent_out(h[:, -n_lat:]).transpose(1, 2).reshape(x_tau.shape)
        v_layout = self.layout_out(h[:, -n_lat - 1:-n_lat])
        return v_shape, v_layout

    @torch.no_grad()
    def sample(self, cond: dict, seeds: list[int], sampler_cfg: DictConfig):
        """Euler ODE from noise (one seed per sample) with classifier-free guidance."""
        device = next(iter(cond.values())).device
        x = torch.stack([torch.randn(self.latent_shape, generator=torch.Generator().manual_seed(s))
                         for s in seeds]).to(device)
        lay = torch.stack([torch.randn(self.layout_shape, generator=torch.Generator().manual_seed(s + 1))
                           for s in seeds]).to(device)
        null, steps, scale = self.null_condition(cond), int(sampler_cfg.steps), float(sampler_cfg.cfg_scale)
        for i in range(steps):
            tau = torch.full((len(seeds),), i / steps, device=device)
            v_c, l_c = self.velocity(x, tau, cond, lay)
            v_u, l_u = self.velocity(x, tau, null, lay)
            x = x + (v_u + scale * (v_c - v_u)) / steps
            lay = lay + (l_u + scale * (l_c - l_u)) / steps
        return x, lay

    @torch.no_grad()
    def zero_shot_layout(self, rgb: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Stand-in for SAM 3D's own zero-shot layout: zeros."""
        return torch.zeros(len(rgb), *self.layout_shape, device=rgb.device)

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


def build_sam3d(cfg: DictConfig, device: torch.device) -> nn.Module:
    """FakeSam3D if cfg.fake_sam3d, else the real (frozen) SAM 3D Geometry model."""
    if cfg.fake_sam3d:
        model = FakeSam3D(d=cfg.adapter.d_cond,
                          activation_checkpointing=cfg.sam3d.activation_checkpointing)
    else:
        model = Sam3DGeometry(cfg)
    return model.to(device)
