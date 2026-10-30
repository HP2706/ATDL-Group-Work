"""One typed configuration for generation, training and evaluation. No project YAML."""
from __future__ import annotations
import ast
from datetime import datetime
import json
import os
from pathlib import Path
from typing import Any, Literal, get_origin, get_type_hints
import chz
from video3d.data.generation.config import Config as GenerationConfig

@chz.chz
class PathsConfig:
    data_root: str = 'data/dataset'
    output_root: str = 'outputs'
    scratch_root: str = 'scratch'
    cache_root: str = 'cache'
    sam3d_checkpoint_dir: str = 'checkpoints/sam3d'
    hf_home: str = '~/.cache/huggingface'


@chz.chz
class CropConfig:
    scale: float = 1.25
    min_side: int = 128
    size: int = 256
    jitter_center: float = 0.1
    jitter_scale: list[float] = chz.field(default_factory=lambda: [0.9, 1.1])


@chz.chz
class DataConfig:
    train_objects: int = 24
    val_objects: int = 4
    test_objects: int = 4
    clips_per_object: int = 2
    pilot_train_objects: int = 2
    pilot_val_objects: int = 2
    train_split: str = 'train'
    val_asset_frac: float = 0.03
    eval_split: str = 'eval'
    stage_to_scratch: bool = False
    num_workers: int = 4
    frame_stride: int = 2
    num_subset_frames: int = 16
    clip_lengths: list[int] = chz.field(default_factory=lambda: [1, 4, 8, 16])
    clip_length_probs: list[float] = chz.field(default_factory=lambda: [0.15, 0.15, 0.2, 0.5])
    p_occluded_bias: float = 0.5
    min_visible_amodal_frac: float = 0.02
    crop: CropConfig = chz.field(default_factory=CropConfig)
    source: str = 'synthetic'
    mask_threshold: float = 0.5
    scanned_strides: list[int] = chz.field(default_factory=lambda: [1, 2, 4])
    prepared: str = 'data/navi-prepared'
    video_input: str = 'object_crop'
    frames: int = 16
    image_size: int = 384


@chz.chz
class VjepaConfig:
    name: str = 'vjepa2_1_vit_large_384'
    weights_url: str = 'https://dl.fbaipublicfiles.com/vjepa2'
    image_size: int = 256
    layers: str = 'last'
    dtype: str = 'bfloat16'
    video_lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 8.0


@chz.chz
class Dinov2Config:
    image_size: int = 224
    pair_average: bool = False


@chz.chz
class EncoderConfig:
    kv_source: str = 'vjepa'
    vjepa: VjepaConfig = chz.field(default_factory=VjepaConfig)
    dinov2: Dinov2Config = chz.field(default_factory=Dinov2Config)


@chz.chz
class AdapterConfig:
    d_cond: int = 1024
    d_vid: int = 1024
    n_blocks: int = 2
    heads: int = 16
    ffn_ratio: float = 2.0
    max_dt: int = 7
    use_object_embedding: bool = True
    use_time_embedding: bool = True
    query_sets: list[str] = chz.field(default_factory=lambda: [])
    strategy: str = 'fusion'
    pooler_queries: int = 64
    pooler_depth: int = 1
    foreground_threshold: float = 0.5


@chz.chz
class Sam3DConfig:
    hf_repo: str = 'facebook/sam-3d-objects'
    activation_checkpointing: bool = True
    phase_a_layout: str = 'noise'
    lora: bool = False
    lora_rank: int = 8
    lora_alpha: float = 8.0
    checkpoint_revision: str = '2e73555018d2741ccd486e56c24fac41155a1dc6'


@chz.chz
class SamplerConfig:
    mode: str = 'shortcut'
    steps: int = 4
    cfg_scale: float | None = None


@chz.chz
class TargetConfig:
    canonical_permutation: list[list[int]] = chz.field(default_factory=lambda: [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    voxel_fill: str = 'surface'


@chz.chz
class LossConfig:
    rotation_weight: float = 0.0
    tau_dist: str = 'logit_normal'
    tau_mean: float = 0.0
    tau_std: float = 1.0


@chz.chz
class OptimConfig:
    lr: float = 0.0001
    betas: list[float] = chz.field(default_factory=lambda: [0.9, 0.999])
    weight_decay: float = 0.01
    warmup_steps: int = 500
    min_lr: float = 1e-06
    grad_clip: float = 1.0
    ema_decay: float = 0.999
    gate_lr_mult: float = 1.0


@chz.chz
class TrainConfig:
    epochs: int = 0
    batch_size: int = 2
    effective_batch_size: int = 32
    max_steps: int = 1000
    bf16: bool = True
    log_every: int = 10
    ckpt_every: int = 1000
    keep_ckpts: int = 0
    resume: str = 'auto'
    benchmark_steps: int = 20
    max_seconds: int = 3600


@chz.chz
class ValConfig:
    epoch_media_samples: int = 0
    loss_every: int = 1000
    loss_num_samples: int = 200
    tau_grid: list[float] = chz.field(default_factory=lambda: [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    sample_every: int = 5000
    sample_num_objects: int = 100


@chz.chz
class MetricsConfig:
    device: str = "auto"
    max_points: int = 20000
    icp_iterations: int = 100
    icp_scale: bool = True
    fscore_thresholds: list[float] = chz.field(default_factory=lambda: [0.02, 0.05])


@chz.chz
class EvalConfig:
    checkpoint_weights: Literal["ema", "current"] = "ema"
    video_diagnostics: bool = False
    video_controls: list[str] = chz.field(default_factory=lambda: ["correct", "repeated", "wrong", "disabled"])
    k: int = 16
    middle_frame: int = 16
    flush_every: int = 20
    metrics: MetricsConfig = chz.field(default_factory=MetricsConfig)


@chz.chz
class AnalysisConfig:
    bootstrap_resamples: int = 2000
    visible_frac_bins: list[float] = chz.field(default_factory=lambda: [0.1, 0.4, 0.7, 1.0])
    arc_span_bins: list[int] = chz.field(default_factory=lambda: [10, 30, 50, 70])


@chz.chz
class LoggingConfig:
    tensorboard: bool = False
    wandb: bool = False
    project: str = 'atdl-video-to-3d'
    entity: str = 'hprjdk'
    media_samples: int = 2


@chz.chz
class ProvenanceConfig:
    git_head: str = ''
    sam_commit: str = 'f91db411c50efee93d8db7aeb323885650f6f722'
    video_commit: str = '204698b45b3712590f06245fbfba32d3be539812'


@chz.chz
class ExperimentConfig:
    generation: GenerationConfig = chz.field(default_factory=GenerationConfig)
    condition: str = 'base'
    env: str = 'local'
    run_name: str = ''
    seed: int = 0
    fake_sam3d: bool = False
    device: str = 'auto'
    paths: PathsConfig = chz.field(default_factory=PathsConfig)
    data: DataConfig = chz.field(default_factory=DataConfig)
    encoder: EncoderConfig = chz.field(default_factory=EncoderConfig)
    adapter: AdapterConfig = chz.field(default_factory=AdapterConfig)
    sam3d: Sam3DConfig = chz.field(default_factory=Sam3DConfig)
    sampler: SamplerConfig = chz.field(default_factory=SamplerConfig)
    target: TargetConfig = chz.field(default_factory=TargetConfig)
    loss: LossConfig = chz.field(default_factory=LossConfig)
    optim: OptimConfig = chz.field(default_factory=OptimConfig)
    train: TrainConfig = chz.field(default_factory=TrainConfig)
    val: ValConfig = chz.field(default_factory=ValConfig)
    eval: EvalConfig = chz.field(default_factory=EvalConfig)
    analysis: AnalysisConfig = chz.field(default_factory=AnalysisConfig)
    logging: LoggingConfig = chz.field(default_factory=LoggingConfig)
    provenance: ProvenanceConfig = chz.field(default_factory=ProvenanceConfig)


def flatten(values: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            result.update(flatten(value, name))
        else:
            if prefix == "generation" and get_origin(get_type_hints(GenerationConfig).get(key)) is tuple:
                value = tuple(value)
            result[name] = value
    return result


def replace_config(cfg: ExperimentConfig, **overrides: Any) -> ExperimentConfig:
    values = flatten(chz.asdict(cfg))
    values.update(overrides)
    return chz.Blueprint(ExperimentConfig).apply(values).make()


def load_config(condition: str | None = None, env: str | None = None,
                overrides: list[str] | None = None) -> ExperimentConfig:
    name = condition or "base"
    presets: dict[str, dict[str, Any]] = {
        "base": {}, "c1": {"encoder.kv_source": "dinov2_ref"},
        "c2": {"encoder.kv_source": "dinov2_clip", "encoder.dinov2.pair_average": True},
        "c3": {}, "c3_nomask": {"adapter.use_object_embedding": False},
        "pooler": {"adapter.strategy": "pooler", "data.video_input": "full_scene", "data.crop.size": 384, "encoder.vjepa.image_size": 384},
        "dino_pooler": {"adapter.strategy": "pooler", "encoder.kv_source": "dinov2_frames",
                        "encoder.dinov2.pair_average": False, "adapter.max_dt": 15},
    }
    if name not in presets:
        raise ValueError(f"Unknown condition {name}; choose from {list(presets)}")
    environment = env or os.environ.get("VIDEO3D_ENV", "local")
    if environment not in {"local", "ucloud", "gcp"}:
        raise ValueError(f"Unknown environment {environment}")
    path_root = "/work/video3d" if environment == "ucloud" else "."
    values: dict[str, Any] = {
        "condition": name, "env": environment,
        "run_name": f"{name}_seed0-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        "paths.data_root": os.environ.get("DATA_ROOT", f"{path_root}/data/dataset"),
        "paths.output_root": os.environ.get("OUTPUT_ROOT", f"{path_root}/outputs"),
        "paths.scratch_root": os.environ.get("SCRATCH_ROOT", f"{path_root}/scratch"),
        "paths.cache_root": os.environ.get("CACHE_ROOT", f"{path_root}/cache"),
        "paths.hf_home": str(Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()),
        "paths.sam3d_checkpoint_dir": os.environ.get("SAM3D_CHECKPOINT_DIR", f"{path_root}/checkpoints/sam3d"),
        "data.num_workers": 0 if environment == "local" else 4,
        **presets[name],
    }
    text_values: dict[str, str] = {}
    for item in overrides or []:
        key, separator, value = item.partition("=")
        if not separator:
            raise ValueError(f"Expected key=value, got {item}")
        if value.lower() in {"true", "false"}:
            values[key] = value.lower() == "true"
        elif value.startswith("["):
            # Accept both Python/JSON lists and the old unquoted list spelling.
            values[key] = [x.strip().strip("\"'") for x in value[1:-1].split(",")] if key == "adapter.query_sets" else ast.literal_eval(value)
        else:
            text_values[key] = value
    blueprint = chz.Blueprint(ExperimentConfig).apply(values)
    cfg = blueprint.apply_from_argv([f"{k}={v}" for k, v in text_values.items()]).make()
    if "run_name" not in text_values:
        cfg = chz.replace(cfg, run_name=f"{name}_seed{cfg.seed}-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
    if cfg.adapter.strategy not in {"fusion", "pooler"}:
        raise ValueError("adapter.strategy must be fusion or pooler")
    if cfg.train.batch_size <= 0 or cfg.train.effective_batch_size <= 0 or cfg.train.effective_batch_size % cfg.train.batch_size:
        raise ValueError("effective_batch_size must be a positive multiple of batch_size")
    if cfg.train.max_steps <= 0 or cfg.train.max_seconds <= 0 or cfg.optim.lr <= 0:
        raise ValueError("steps, wall time and learning rate must be positive")
    if cfg.train.epochs < 0 or cfg.val.epoch_media_samples < 0:
        raise ValueError("epochs and per-epoch media counts must be nonnegative")
    if cfg.data.source not in {"synthetic", "navi", "gso", "omniobject3d"}:
        raise ValueError("data.source must be synthetic, navi, gso or omniobject3d")
    return cfg


def save_config(cfg: ExperimentConfig, run_dir: str | Path) -> Path:
    path = Path(run_dir) / "config.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(chz.asdict(cfg), indent=2, allow_nan=False))
    return path


def read_config(path: str | Path) -> ExperimentConfig:
    return chz.Blueprint(ExperimentConfig).apply(flatten(json.loads(Path(path).read_text()))).make()


def apply_process_env(cfg: ExperimentConfig) -> None:
    os.environ.setdefault("HF_HOME", cfg.paths.hf_home)
    os.environ.setdefault("LIDRA_SKIP_INIT", "1")
    os.environ.setdefault("ATTN_BACKEND", "sdpa")


def config_dict(cfg: ExperimentConfig) -> dict[str, Any]:
    """Canonical JSON-compatible representation, including tuple-valued generation settings."""
    return json.loads(json.dumps(chz.asdict(cfg)))
