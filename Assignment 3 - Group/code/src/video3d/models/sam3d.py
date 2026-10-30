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
from video3d.models.conditioning import ConditioningAdapter, VideoFeatures

log = logging.getLogger("video3d")
STAGE2_MODELS = ("slat_generator", "slat_decoder_gs", "slat_decoder_gs_4", "slat_decoder_mesh")


def apply_adapter(cond: dict[str, torch.Tensor], adapter: ConditioningAdapter,
                  tokens: torch.Tensor, coverage: torch.Tensor, dt: torch.Tensor,
                  query_sets: list[str], kv_padding_mask: torch.Tensor | None = None,
                  video_disabled: bool = False, diagnostics: dict[str, typing.Any] | None = None,
                  mask_patches: torch.Tensor | None = None,
                  positions: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
    """Compatibility entrypoint; every strategy owns its conditioning assembly."""
    video = VideoFeatures(tokens=tokens, coverage=coverage, dt=dt, padding=kv_padding_mask,
                          mask_patches=mask_patches, positions=positions)
    return adapter.condition(cond, video, query_sets, video_disabled, diagnostics)


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
        if not cfg.eval.refined_mesh and cfg.train.stage != 'stage2':
            for name in STAGE2_MODELS:
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
        self.sampling_rng_state: torch.Tensor | None = None
        self.requires_grad_(False)
        self.eval()
        if cfg.sam3d.lora and cfg.train.stage == 'stage1':
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
        self.stage1_cfg = None
        self.stage1_adapter = None
        self.stage1_encoder = None
        if cfg.train.stage == 'stage2':
            from video3d.stage2 import initialize_stage2
            initialize_stage2(self, cfg, device)

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
            with p.device:
                pointmap = p.compute_pointmap(rgba)["pointmap"]
                inputs = p.preprocess_image(rgba, p.ss_preprocessor, pointmap=pointmap)
            with p.device, self._autocast():
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
        if 'coords' in layout_in:
            backbone = self.pipeline.models['slat_generator'].reverse_fn.backbone
            generator = self.pipeline.models['slat_generator']
            options = {'d': torch.zeros_like(tau)} if backbone.is_shortcut_model else {}
            with torch.autocast('cuda', dtype=self.pipeline.dtype):
                prediction = backbone(x_tau, tau.float() * generator.time_scale,
                                      torch.cat(list(cond.values()), 1), layout_in['coords'].cpu().numpy(), **options)
            return prediction, {}
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
                    self.sampling_rng_state = torch.cuda.get_rng_state(self.device_)
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
        """SAM 3D's own DINOv2 for reference and per-frame controls."""
        from sam3d_objects.model.backbone.dit.embedder.dino import Dino
        for module in self.embedder.module_list:
            if isinstance(module, Dino):
                return module.backbone
        raise AssertionError("no Dino embedder in the condition embedder")

    @torch.no_grad()
    def sparse_structure(self, z: torch.Tensor, layout: dict[str, torch.Tensor]) -> dict[str, typing.Any]:
        """Use the same native pruning/downsampling as the baseline Stage-1 pipeline."""
        from sam3d_objects.pipeline.inference_pipeline import prune_sparse_structure, downsample_sparse_structure
        coords = torch.argwhere(self.decode_latent(z))[:, [0, 1, 2, 3]].int()
        original = coords.clone()
        if self.pipeline.downsample_ss_dist > 0:
            coords = prune_sparse_structure(coords, max_neighbor_axes_dist=self.pipeline.downsample_ss_dist)
        coords, factor = downsample_sparse_structure(coords)
        return {"shape":z, **layout, "coords":coords, "coords_original":original,
                "downsample_factor":factor}

    @torch.no_grad()
    def refine(self, z: torch.Tensor, layout: dict[str, torch.Tensor], rgb: torch.Tensor,
               mask: torch.Tensor, seed: int, steps: int,
               condition: dict[str, torch.Tensor] | None = None) -> dict[str, typing.Any]:
        """Native Stage 2 from our sampled Stage 1; never replace it with zero-shot geometry."""
        pipeline = self.pipeline
        # Native sample_slat owns its autocast. FlexiCubes mesh extraction must
        # run outside it, as in pipeline.run, to avoid Float/Half index_add errors.
        with self.device_, torch.random.fork_rng(devices=[self.device_]):
            image = pipeline.merge_image_and_mask(rgb[0].cpu().numpy(), mask[0].cpu().numpy().astype(np.uint8) * 255)
            pointmap = pipeline.compute_pointmap(image)["pointmap"]
            inputs = pipeline.preprocess_image(image, pipeline.ss_preprocessor, pointmap=pointmap)
            coarse = self.sparse_structure(z, layout)
            coords, factor = coarse['coords'], coarse['downsample_factor']
            coarse.update(pipeline.pose_decoder(coarse, scene_scale=inputs.get("pointmap_scale"),
                                                scene_shift=inputs.get("pointmap_shift")))
            coarse["scale"] = coarse["scale"] * factor
            assert self.sampling_rng_state is not None, 'Call Stage-1 sample before refinement'
            torch.cuda.set_rng_state(self.sampling_rng_state, self.device_)
            if condition is None:
                slat = pipeline.sample_slat(pipeline.preprocess_image(image, pipeline.slat_preprocessor),
                                            coords, inference_steps=steps, use_distillation=False)
            else:
                from video3d.stage2 import sample_refinement
                slat = sample_refinement(self, condition, coords, steps)
            outputs = pipeline.decode_slat(slat, ["mesh", "gaussian"])
            outputs['slat_features'] = slat.feats.detach().cpu()
            outputs['slat_coordinates'] = slat.coords.detach().cpu()
            outputs = pipeline.postprocess_slat_output(outputs, False, False, True)
            return {**coarse, **outputs, 'pointmap':pointmap}


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
