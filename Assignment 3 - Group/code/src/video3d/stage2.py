"""Stage-2 hooks for the existing trainer, checkpoint writer and refined evaluator."""
from __future__ import annotations

import hashlib
from contextlib import contextmanager
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch

from video3d.config import ExperimentConfig, read_config
from video3d.data.stage2 import audit_targets, file_hash, load_target
from video3d.loss import flow_loss


def lora_settings(cfg: ExperimentConfig) -> tuple[bool, int, float]:
    """Stage-2 overrides are independent of the pinned Stage-1 checkpoint settings."""
    return (cfg.sam3d.lora if cfg.stage2.lora is None else cfg.stage2.lora,
            cfg.sam3d.lora_rank if cfg.stage2.lora_rank is None else cfg.stage2.lora_rank,
            cfg.sam3d.lora_alpha if cfg.stage2.lora_alpha is None else cfg.stage2.lora_alpha)


def validate_training(cfg: ExperimentConfig) -> None:
    if cfg.train.stage != 'stage2':
        return
    if cfg.fake_sam3d:
        raise ValueError('Stage-2 uses the actual sparse model; FakeSam3D is Stage-1-only')
    if not cfg.stage2.target_root:
        raise ValueError('Stage-2 training requires stage2.target_root with verified source-mesh latent targets')
    if cfg.val.epoch_media_samples:
        raise ValueError('Stage-2 uses heavy refined evaluation; coarse epoch previews/full-validation cannot score its changes')
    enabled, rank, alpha = lora_settings(cfg)
    if enabled and (rank <= 0 or alpha <= 0):
        raise ValueError('Stage-2 LoRA rank and alpha must be positive')
    if not enabled and cfg.adapter.strategy == 'native':
        raise ValueError('Stage-2 native conditioning needs sam3d.lora=True to have trainable parameters')


def initialize_stage2(model: Any, cfg: ExperimentConfig, device: torch.device) -> None:
    """Restore and freeze Stage 1 before installing independently trained Stage-2 adapters."""
    from video3d.adapters import inject_lora
    from video3d.checkpoint import extra_parameters, restore_extra
    from video3d.evaluate import load_run
    from video3d.models.encoders import build_kv_encoder

    if cfg.stage2.stage1_run:
        run = Path(cfg.stage2.stage1_run)
        base = read_config(run / 'config.json')
        if base.train.stage != 'stage1' or base.encoder.vjepa.video_lora:
            raise ValueError('Stage-1 handoff requires a Stage-1 checkpoint with a frozen video encoder')
        if base.sam3d.checkpoint_revision != cfg.sam3d.checkpoint_revision:
            raise ValueError('Stage-1 and Stage-2 SAM checkpoint revisions differ')
        if not cfg.stage2.stage1_checkpoint:
            raise ValueError('Pin stage2.stage1_checkpoint explicitly; do not follow a moving latest pointer')
        if base.sam3d.lora:
            inject_lora(model.backbone, r'blocks\.\d+\.cross_attn\.shape\.(to_q|to_kv|to_out)',
                        base.sam3d.lora_rank, base.sam3d.lora_alpha)
        base, adapter, _ = load_run(base, str(run), device, cfg.stage2.stage1_checkpoint)
        encoder = build_kv_encoder(base, model, device)
        restore_extra(extra_parameters(encoder, model), adapter.extra_ema_state)
        model.stage1_cfg, model.stage1_adapter, model.stage1_encoder = base, adapter, encoder
    # InferencePipeline is not an nn.Module. Register the sparse generator so
    # optimizer/EMA/checkpoint discovery sees its adapters, then freeze its base.
    model.slat_generator = model.pipeline.models['slat_generator']
    model.requires_grad_(False)
    model.pipeline.condition_embedders['slat_condition_embedder'].requires_grad_(False)
    backbone = model.pipeline.models['slat_generator'].reverse_fn.backbone
    enabled, rank, alpha = lora_settings(cfg)
    model.stage2_lora_modules = []
    if enabled:
        model.stage2_lora_modules = inject_lora(backbone, r'blocks\.\d+\.cross_attn\.(to_q|to_kv|to_out)',
                    rank, alpha)
    for block in backbone.blocks:
        block.use_checkpoint = bool(cfg.sam3d.activation_checkpointing)


@torch.no_grad()
def condition(model: Any, rgb: torch.Tensor, mask: torch.Tensor) -> dict[str, torch.Tensor]:
    pipeline = model.pipeline
    embedder = pipeline.condition_embedders['slat_condition_embedder']
    names = [name for _, info in embedder.embedder_list for name, _ in info]
    lengths: list[int] = []
    handles = [module.register_forward_hook(lambda mod, inp, out: lengths.append(out.shape[1]))
               for module in embedder.module_list]
    tokens = []
    try:
        for b in range(len(rgb)):
            image = pipeline.merge_image_and_mask(rgb[b].cpu().numpy(), mask[b].cpu().numpy().astype(np.uint8) * 255)
            with pipeline.device:
                inputs = pipeline.preprocess_image(image, pipeline.slat_preprocessor)
            with pipeline.device, torch.autocast('cuda', dtype=pipeline.dtype):
                lengths.clear()
                args, _ = pipeline.get_condition_input(embedder, inputs, pipeline.slat_condition_input_mapping)
            tokens.append(args[0])
    finally:
        for handle in handles:
            handle.remove()
    if len(lengths) != len(names) or sum(lengths) != tokens[0].shape[1]:
        raise ValueError('Stage-2 conditioning token boundaries changed')
    return dict(zip(names, torch.cat(tokens).split(lengths, dim=1)))


def training_loss(model: Any, cond: dict[str, torch.Tensor], batch: list[dict[str, Any]],
                  cfg: ExperimentConfig, device: torch.device, generator: torch.Generator,
                  taus: torch.Tensor) -> torch.Tensor:
    """Equal weight per object, independent of the number of its occupied surface cells.

    The native sparse wrapper accepts one sparse example at a time. Accumulating the
    per-object losses retains the trainer's existing effective-batch normalization.
    """
    losses = []
    for i, sample in enumerate(batch):
        target = load_target(cfg, sample['info']['asset_id'])
        mean, std = model.pipeline.slat_mean.to(device), model.pipeline.slat_std.to(device)
        x1 = ((target.features.to(device) - mean) / std)[None]
        x0 = torch.randn(x1.shape, generator=generator, device=device)
        coords = torch.cat([torch.zeros(len(target.coords), 1, dtype=torch.int32), target.coords], 1)
        losses.append(flow_loss(model, {k: v[i:i+1] for k, v in cond.items()}, x1,
                                taus[i:i+1], x0, {'coords': coords}))
    return torch.stack(losses).mean()


@torch.no_grad()
def coarse_condition(model: Any, sample: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    from video3d.train import adapted_condition, reference_images
    if model.stage1_cfg is None:
        return model.condition(*reference_images([sample], device))
    return adapted_condition(model, model.stage1_encoder, model.stage1_adapter, [sample], model.stage1_cfg, device)


@torch.no_grad()
def sample_refinement(model: Any, cond: dict[str, torch.Tensor], coords: torch.Tensor, steps: int) -> Any:
    from sam3d_objects.model.backbone.tdfy_dit.modules import sparse as sp
    generator = model.pipeline.models['slat_generator']
    with refinement_sampler(generator, steps, model.pipeline.slat_cfg_strength):
        with torch.autocast('cuda', dtype=model.pipeline.dtype):
            features = generator((1, len(coords), 8), model.device_, torch.cat(list(cond.values()), 1),
                                 coords.cpu().numpy())[0]
            output = sp.SparseTensor(coords=coords, feats=features)
            return output * model.pipeline.slat_std.to(model.device_) + model.pipeline.slat_mean.to(model.device_)


@contextmanager
def refinement_sampler(generator: Any, steps: int, strength: float) -> Iterator[None]:
    """Plain FlowMatching has no shortcut state; the backbone declares that contract."""
    guidance = generator.reverse_fn
    previous = generator.inference_steps, guidance.strength
    shortcut = guidance.backbone.is_shortcut_model
    previous_shortcut = generator.no_shortcut if shortcut else False
    generator.inference_steps, guidance.strength = steps, strength
    if shortcut:
        generator.no_shortcut = True
    try:
        yield
    finally:
        generator.inference_steps, guidance.strength = previous
        if shortcut:
            generator.no_shortcut = previous_shortcut


def audit_training_targets(cfg: ExperimentConfig, train_ds: Any, val_ds: Any | None) -> None:
    ids = [item['asset_id'] if 'asset_id' in item else ds.clip(item['clip_id']).info['objects'][item['k']-1]['asset_id']
           for ds in [train_ds, val_ds] if ds is not None for item in ds.index]
    audit_targets(cfg, ids)


def stage2_identity(cfg: ExperimentConfig) -> str:
    manifest = Path(cfg.stage2.target_root) / 'manifest.json'
    signature = [file_hash(manifest)]
    if cfg.stage2.stage1_run:
        signature.extend([file_hash(Path(cfg.stage2.stage1_run) / 'config.json'),
                          file_hash(Path(cfg.stage2.stage1_checkpoint))])
    return hashlib.sha256('|'.join(signature).encode()).hexdigest()
