"""Frozen SAM 3D Objects Geometry model (Stage 1 only), wrapped without changing its source.

Interface (shared with FakeSam3D in fake_sam3d.py):
  condition(rgb, mask)               -> dict[name -> (B, N, d)] conditioning token sets (no grad)
  velocity(x_tau, tau, cond, layout) -> (v_shape, dict of v_layout); d = 0 (flow matching)
  sample(cond, seeds, sampler_cfg)   -> (shape latent, layout dict); CFG null branch = zeros
  encode_voxels(vox) / decode_latent(z)  64^3 bool <-> shape latent (no normalisation)
  zero_shot_layout(rgb, mask)        -> layout dict sampled without the adapter
  latent_shape, layout_shape         per-sample shapes; layout_shape is {name: shape}
See docs/sam3d_internals.md for where each behaviour comes from in the SAM 3D code.
"""

from __future__ import annotations
import typing

import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from video3d.config import ExperimentConfig, SamplerConfig

log = logging.getLogger("video3d")
STAGE2_MODELS = ("slat_generator", "slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh")


def apply_adapter(cond: dict, adapter: nn.Module, tokens: typing.Any, coverage: typing.Any, dt: typing.Any, query_sets: typing.Any,
                  kv_padding_mask: typing.Any=None, video_disabled: bool = False,
                  diagnostics: dict[str, typing.Any] | None = None) -> dict:
    """Return a new cond dict where only `query_sets` are replaced by the adapter output."""
    from video3d.models.pooler import MaskPooler
    query_sets = query_sets or [name for name in cond if ("rgb" in name or "image" in name) and "mask" not in name and "pointmap" not in name]
    if not query_sets:
        raise ValueError(f"Set adapter.query_sets to RGB conditioning names: {list(cond)}")
    missing = [name for name in query_sets if name not in cond]
    assert not missing, f"query sets {missing} not in conditioning sets {list(cond)}"
    if isinstance(adapter, MaskPooler):
        if diagnostics is not None:
            diagnostics["attention_available"] = False
            diagnostics["reason"] = "Pooler replaces conditioning tokens; gated cross-attention diagnostics do not apply"
        if video_disabled:
            return cond
        return adapter.condition(cond, tokens, coverage, query_sets, kv_padding_mask)
    sizes = [cond[name].shape[1] for name in query_sets]
    c_sel = torch.cat([cond[name] for name in query_sets], dim=1).float()
    if diagnostics is None and not video_disabled:
        out = adapter(c_sel, tokens, coverage, dt, kv_padding_mask)
    else:
        out = adapter(c_sel, tokens, coverage, dt, kv_padding_mask,
                      video_disabled=video_disabled, diagnostics=diagnostics)
    return {**cond, **dict(zip(query_sets, out.split(sizes, dim=1)))}


class Sam3DGeometry(nn.Module):
    """The official InferencePipelinePointMap, reduced to its Stage-1 modules and frozen."""

    def __init__(self, cfg: ExperimentConfig, device: torch.device) -> None:
        super().__init__()
        os.environ.setdefault("LIDRA_SKIP_INIT", "1")
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        from hydra.utils import instantiate

        ckpt_dir = Path(cfg.paths.sam3d_checkpoint_dir)
        pcfg = OmegaConf.load(ckpt_dir / "pipeline.yaml")
        pcfg.workspace_dir, pcfg.device = str(ckpt_dir), str(device)
        pcfg.rendering_engine, pcfg.compile_model = "pytorch3d", False
        if pcfg.get("ss_encoder_ckpt_path") is None and (ckpt_dir / "ss_encoder.ckpt").is_file():
            pcfg.ss_encoder_config_path, pcfg.ss_encoder_ckpt_path = "ss_encoder.yaml", "ss_encoder.ckpt"
        self.pipeline_cfg = pcfg
        self.pipeline = instantiate(pcfg)
        for name in STAGE2_MODELS:                       # never run Texture & Refinement
            if name in self.pipeline.models:
                del self.pipeline.models[name]
        self.pipeline.condition_embedders["slat_condition_embedder"] = None
        torch.cuda.empty_cache()

        self.embedder = self.pipeline.condition_embedders["ss_condition_embedder"]
        self.generator = self.pipeline.models["ss_generator"]
        self.decoder = self.pipeline.models["ss_decoder"]
        self.encoder = self.pipeline.models["ss_encoder"]
        self.backbone = self.generator.reverse_fn.backbone
        self.dtype = self.pipeline.shape_model_dtype
        self.device_ = torch.device(device)
        self.sampler_cfg = cfg.sampler
        self.requires_grad_(False)
        self.eval()
        if cfg.sam3d.lora:
            from video3d.adapters import inject_lora
            inject_lora(self.backbone, r"blocks\.\d+\.cross_attn\.shape\.(to_q|to_kv|to_out)",
                        cfg.sam3d.lora_rank, cfg.sam3d.lora_alpha)
        for block in self.backbone.blocks:
            block.use_checkpoint = bool(cfg.sam3d.activation_checkpointing)

        assert self.embedder is not None, "pipeline has no ss_condition_embedder"
        assert self.embedder.compression_projection_multiplier == 0,\
            "condition sets are fused along channels; cannot split them into token sets"
        self.set_names = [name for _, info in self.embedder.embedder_list for name, _ in info]
        mapping = self.backbone.latent_mapping
        assert "shape" in mapping, f"no 'shape' latent in {list(mapping)}"
        shape_of = lambda m: (m.pos_emb.shape[0], m.input_layer.in_features)
        self.latent_shape = shape_of(mapping["shape"])
        self.layout_shape = {k: shape_of(m) for k, m in mapping.items() if k != "shape"}
        self._lengths: list[int] = []
        for module in self.embedder.module_list:
            module.register_forward_hook(lambda mod, inp, out: self._lengths.append(out.shape[1]))

    def _autocast(self) -> typing.Any:
        return torch.autocast(device_type="cuda", dtype=self.dtype)

    # ---- conditioning ----

    @torch.no_grad()
    def condition(self, rgb: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
        """rgb (B, H, W, 3) uint8, mask (B, H, W) bool -> named conditioning token sets."""
        p = self.pipeline
        tokens = []
        for b in range(len(rgb)):
            rgba = p.merge_image_and_mask(rgb[b].cpu().numpy(), mask[b].cpu().numpy().astype(np.uint8) * 255)
            with p.device, self._autocast():
                pointmap = p.compute_pointmap(rgba)["pointmap"]
                inputs = p.preprocess_image(rgba, p.ss_preprocessor, pointmap=pointmap)
                self._lengths.clear()
                args, _ = p.get_condition_input(self.embedder, inputs, p.ss_condition_input_mapping)
            tokens.append(args[0])
        lengths = list(self._lengths)
        assert len(lengths) == len(self.set_names) and sum(lengths) == tokens[0].shape[1],\
            f"token-set split failed: {lengths} for {self.set_names}, total {tokens[0].shape[1]}"
        return dict(zip(self.set_names, torch.cat(tokens).split(lengths, dim=1)))

    # ---- flow model ----

    def velocity(self, x_tau: typing.Any, tau: typing.Any, cond: dict, layout_in: dict) -> typing.Any:
        """Backbone velocity at flow time τ ∈ [0, 1] with step size d = 0 (no CFG, no dropout)."""
        t = tau.float() * self.generator.time_scale
        kwargs = {"d": torch.zeros_like(t)} if self.backbone.is_shortcut_model else {}
        out = self.backbone({"shape": x_tau, **layout_in}, t, torch.cat(list(cond.values()), 1), **kwargs)
        v_shape = out.pop("shape")
        return v_shape, out

    @torch.no_grad()
    def sample(self, cond: dict, seeds: list[int], sampler_cfg: SamplerConfig) -> typing.Any:
        """Stage-1 sampling with the pipeline's own generator; one seed per sample."""
        gen, cfg_fn = self.generator, self.generator.reverse_fn
        saved = gen.no_shortcut, cfg_fn.strength, cfg_fn.strength_pm, gen.inference_steps
        if sampler_cfg.mode == "shortcut":               # distilled: d = 1/steps, no CFG
            gen.no_shortcut, cfg_fn.strength, cfg_fn.strength_pm = False, 0, 0
        elif sampler_cfg.mode == "flow":                 # d = 0 with CFG (null branch = zeros)
            assert cfg_fn.unconditional_handling == "add_flag" and self.backbone.force_zeros_cond,\
                "CFG null branch would not be zeros; see docs/sam3d_internals.md item 3"
            gen.no_shortcut = True
            cfg_fn.strength = self.pipeline.ss_cfg_strength if sampler_cfg.cfg_scale is None\
                else float(sampler_cfg.cfg_scale)
            cfg_fn.strength_pm = self.pipeline.ss_cfg_strength_pm
        else:
            raise ValueError(f"unknown sampler.mode {sampler_cfg.mode!r}")
        gen.inference_steps = int(sampler_cfg.steps)
        tokens = torch.cat(list(cond.values()), 1)
        shapes = {"shape": (1, *self.latent_shape), **{k: (1, *s) for k, s in self.layout_shape.items()}}
        outs = []
        try:
            for i, seed in enumerate(seeds):
                with torch.random.fork_rng(devices=[self.device_]), self._autocast():
                    torch.manual_seed(seed)
                    outs.append(gen(shapes, self.device_, tokens[i:i + 1]))
        finally:
            gen.no_shortcut, cfg_fn.strength, cfg_fn.strength_pm, gen.inference_steps = saved
        z = torch.cat([o["shape"] for o in outs]).float()
        return z, {k: torch.cat([o[k] for o in outs]).float() for k in self.layout_shape}

    @torch.no_grad()
    def zero_shot_layout(self, rgb: torch.Tensor, mask: torch.Tensor) -> dict:
        """Layout latents of plain SAM 3D on the reference frame (seed 0, eval sampler).

        Not needed in Phase A: shape tokens do not attend to layout tokens (internals item 7).
        """
        _, layout = self.sample(self.condition(rgb, mask), [0] * len(rgb), self.sampler_cfg)
        return layout

    # ---- sparse-structure VAE ----

    @torch.no_grad()
    def encode_voxels(self, vox: torch.Tensor) -> torch.Tensor:
        """(B, 64, 64, 64) bool -> (B, 4096, 8) encoder mean in the generator's token layout."""
        assert self.encoder is not None, "no ss_encoder.ckpt in the checkpoint dir"
        with self._autocast():
            mean = self.encoder(vox.float()[:, None])["mean"]
        return mean.float().flatten(2).transpose(1, 2)

    @torch.no_grad()
    def decode_latent(self, z: torch.Tensor) -> torch.Tensor:
        """(B, 4096, 8) -> (B, 64, 64, 64) bool, thresholding decoder logits at 0 like the pipeline."""
        with self._autocast():
            logits = self.decoder(z.transpose(1, 2).reshape(len(z), 8, 16, 16, 16))
        return logits[:, 0] > 0

    def dino_backbone(self) -> nn.Module:
        """SAM 3D's own DINOv2 (for the C1/C2 controls)."""
        from sam3d_objects.model.backbone.dit.embedder.dino import Dino
        for module in self.embedder.module_list:
            if isinstance(module, Dino):
                return module.backbone
        raise AssertionError("no Dino embedder in the condition embedder")


def build_sam3d(cfg: ExperimentConfig, device: torch.device) -> nn.Module:
    """FakeSam3D if cfg.fake_sam3d, else the real (frozen) SAM 3D Geometry model."""
    if cfg.fake_sam3d:
        from video3d.models.fake_sam3d import FakeSam3D
        model = FakeSam3D(d=cfg.adapter.d_cond,
                           activation_checkpointing=cfg.sam3d.activation_checkpointing).to(device)
        if cfg.sam3d.lora:
            from video3d.adapters import inject_lora
            inject_lora(model, "latent_in", cfg.sam3d.lora_rank, cfg.sam3d.lora_alpha)
        return model
    return Sam3DGeometry(cfg, device)
