from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any
import torch
from torch import nn
from hydra.utils import instantiate
from omegaconf import OmegaConf
from huggingface_hub import snapshot_download
from .adapters import inject_lora, LoRALinear
from .config import ModelConfig


VIDEO_SELECTOR = r"blocks\.\d+\.attn\.(qkv|proj)"
SAM_SELECTOR = r"blocks\.\d+\.cross_attn\.shape\.(to_q|to_kv|to_out)"


def foreground_tokens(foreground: torch.Tensor, threshold: float) -> torch.Tensor:
    if foreground.ndim != 4 or foreground.shape[1] % 2 or foreground.shape[2] % 16 or foreground.shape[3] % 16:
        raise ValueError("Foreground masks must match the two-frame, 16x16 tubelet grid")
    coverage = torch.nn.functional.avg_pool3d(foreground[:, None].float(), kernel_size=(2, 16, 16), stride=(2, 16, 16)).flatten(1)
    selected = coverage >= threshold
    if not selected.any(dim=1).all():
        raise ValueError("No foreground tubelet passes the configured threshold")
    return selected


class VideoConditioning(nn.Module):
    def __init__(self, encoder: nn.Module, video_width: int, condition_width: int, frames: int, pooler: nn.Module | None = None, foreground_threshold: float = 0.5, native_tokens_per_view: int = 1370) -> None:
        super().__init__()
        self.encoder = encoder
        self.projector = nn.Linear(video_width, condition_width)
        self.frames = frames
        self.pooler = pooler
        self.foreground_threshold = foreground_threshold
        self.native_tokens_per_view = native_tokens_per_view

    def forward(self, video: torch.Tensor, reference: torch.Tensor, foreground: torch.Tensor | None = None) -> torch.Tensor:
        features = self.encoder(video)
        if not isinstance(features, torch.Tensor) or features.shape[-1] != self.projector.in_features:
            raise ValueError("Unexpected V-JEPA feature contract")
        if self.pooler is None:
            spatial = features.reshape(features.shape[0], self.frames // 2, -1, features.shape[-1]).mean(dim=1)
            projected = self.projector(spatial.float())
            return torch.cat([reference.to(projected.dtype), projected], dim=1)
        if foreground is None:
            raise ValueError("Mask-aware pooling requires aligned foreground masks")
        selected = foreground_tokens(foreground, self.foreground_threshold)
        if selected.shape != features.shape[:2]:
            raise ValueError("Foreground tubelets and V-JEPA tokens disagree")
        if reference.shape[1] != 4 * self.native_tokens_per_view + 2048:
            raise ValueError("Reference conditioning does not match the pinned native fuser layout")
        # No padding: each sample's pooler sees only its selected object tokens.
        summaries = torch.cat([self.pooler(features[index:index + 1, selected[index]]) for index in range(features.shape[0])], dim=0)
        projected = self.projector(summaries.float())
        # Replace both native RGB streams; retain both mask streams and zero pointmap slots.
        retained = reference[:, 2 * self.native_tokens_per_view:]
        return torch.cat([projected, retained.to(projected.dtype)], dim=1)


def load_component(config: Any, checkpoint: Path, prefix: str = "", state_key: str | None = "state_dict") -> nn.Module:
    config = OmegaConf.create(OmegaConf.to_container(config, resolve=True))
    if "pretrained_ckpt_path" in config:
        del config["pretrained_ckpt_path"]
    model = instantiate(config)
    payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = payload if state_key is None else payload[state_key]
    if prefix:
        state = {name.removeprefix(prefix): tensor for name, tensor in state.items() if name.startswith(prefix)}
    model.load_state_dict(state, strict=True)
    return model.eval().requires_grad_(False)


def bf16_with_float_norms(model: nn.Module) -> None:
    model.bfloat16()
    # Upstream LayerNorm32 casts activations to FP32 before calling LayerNorm.
    for module in model.modules():
        if isinstance(module, (nn.LayerNorm, nn.GroupNorm)):
            module.float()


class NativeSAM(nn.Module):
    """Owns upstream modules; delegates training loss and generation upstream."""
    def __init__(self, config: ModelConfig, frames: int, device: torch.device) -> None:
        super().__init__()
        os.environ["LIDRA_SKIP_INIT"] = "1"
        os.environ["SAM3D_GEOMETRY_ONLY"] = "1"
        sys.path.insert(0, config.sam_source)
        checkpoint_root = Path(snapshot_download(config.sam_weights, revision=config.sam_revision, allow_patterns=[f"checkpoints/ss_{name}.{extension}" for name in ["generator", "encoder", "decoder"] for extension in ["yaml", "ckpt"]], token=os.environ.get("HF_READ_TOKEN") or os.environ.get("HF_TOKEN"))) / "checkpoints"
        self.checkpoint_root = str(checkpoint_root)
        generator_config = OmegaConf.load(checkpoint_root / "ss_generator.yaml")["module"]["generator"]["backbone"]
        self.generator = load_component(generator_config, checkpoint_root / "ss_generator.ckpt", "_base_models.generator.")
        self.decoder = load_component(OmegaConf.load(checkpoint_root / "ss_decoder.yaml"), checkpoint_root / "ss_decoder.ckpt", state_key=None)
        self.encoder = load_component(OmegaConf.load(checkpoint_root / "ss_encoder.yaml"), checkpoint_root / "ss_encoder.ckpt", state_key=None)
        self.encoder.sample_posterior = False
        sys.path.insert(0, config.video_source)
        from src.hub import backbones as video_hub
        video_hub.VJEPA_BASE_URL = config.video_weights_base_url
        video_encoder, predictor = torch.hub.load(config.video_source, config.video_entrypoint, source="local", pretrained=True)
        del predictor
        video_encoder.requires_grad_(False).eval().bfloat16()
        bf16_with_float_norms(self.generator)
        self.generator.reverse_fn.backbone.dtype = torch.bfloat16
        video_encoder.use_activation_checkpointing = True
        pooler: nn.Module | None = None
        if config.conditioning == "mask_pooler":
            from src.models.attentive_pooler import AttentivePooler
            pooler = AttentivePooler(num_queries=config.pooler_queries, embed_dim=config.video_width, num_heads=12, depth=config.pooler_depth, use_activation_checkpointing=True)
        self.conditioning = VideoConditioning(video_encoder, config.video_width, config.condition_width, frames, pooler=pooler, foreground_threshold=config.foreground_threshold, native_tokens_per_view=config.native_tokens_per_view)
        self.generator.reverse_fn.backbone.use_checkpoint = True
        for block in self.generator.reverse_fn.backbone.blocks:
            block.use_checkpoint = True
        self.generator.self_consistency_prob = 0.0
        self.generator.no_shortcut = True
        self.generator.reverse_fn.p_unconditional = 0.0
        self.generator.loss_weights = {key: 1.0 if key == "shape" else 0.0 for key in self.generator.reverse_fn.backbone.latent_mapping}
        self.lora_targets: dict[str, list[str]] = {"video": [], "sam": []}
        if config.video_lora:
            self.lora_targets["video"] = inject_lora(video_encoder, VIDEO_SELECTOR, config.rank, config.alpha)
        if config.sam_lora:
            self.lora_targets["sam"] = inject_lora(self.generator.reverse_fn.backbone, SAM_SELECTOR, config.rank, config.alpha)
        self.to(device)

    def loss(self, video: torch.Tensor, reference: torch.Tensor, targets: dict[str, torch.Tensor], foreground: torch.Tensor | None = None) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        conditioning = self.conditioning(video, reference, foreground)
        return self.generator.loss(targets, conditioning)

    @torch.no_grad()
    def sample_condition(self, conditioning: torch.Tensor, steps: int) -> torch.Tensor:
        self.generator.inference_steps = steps
        dimensions = {key: (conditioning.shape[0], value.pos_emb.shape[0], value.input_layer.in_features) for key, value in self.generator.reverse_fn.backbone.latent_mapping.items()}
        result = self.generator(dimensions, conditioning.device, conditioning)
        latent = result["shape"].permute(0, 2, 1).contiguous().reshape(conditioning.shape[0], 8, 16, 16, 16)
        return self.decoder(latent)

    @torch.no_grad()
    def generate(self, video: torch.Tensor, reference: torch.Tensor, steps: int, foreground: torch.Tensor | None = None) -> torch.Tensor:
        return self.sample_condition(self.conditioning(video, reference, foreground), steps)

    @torch.no_grad()
    def generate_native_image(self, reference: torch.Tensor, steps: int) -> torch.Tensor:
        adapters = [(module, module.scale) for module in self.generator.modules() if isinstance(module, LoRALinear)]
        for module, _ in adapters:
            module.scale = 0.0
        try:
            return self.sample_condition(reference, steps)
        finally:
            for module, scale in adapters:
                module.scale = scale

    @torch.no_grad()
    def encode_geometry(self, occupancy: torch.Tensor) -> dict[str, torch.Tensor]:
        shape = self.encoder(occupancy)["mean"].flatten(2).transpose(1, 2)
        return {key: shape if key == "shape" else torch.zeros((shape.shape[0], value.pos_emb.shape[0], value.input_layer.in_features), device=shape.device) for key, value in self.generator.reverse_fn.backbone.latent_mapping.items()}
