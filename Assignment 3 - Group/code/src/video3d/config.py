from __future__ import annotations

from datetime import datetime
from pathlib import Path
import subprocess
import chz


@chz.chz
class ModelConfig:
    sam_source: str = "/work/dependencies/sam-3d-objects"
    sam_commit: str = "f91db411c50efee93d8db7aeb323885650f6f722"
    sam_weights: str = "facebook/sam-3d-objects"
    sam_revision: str = "2e73555018d2741ccd486e56c24fac41155a1dc6"
    video_source: str = "/work/dependencies/vjepa2"
    video_commit: str = "204698b45b3712590f06245fbfba32d3be539812"
    video_weights_base_url: str = "https://dl.fbaipublicfiles.com/vjepa2"
    video_entrypoint: str = "vjepa2_1_vit_base_384"
    video_width: int = 768
    condition_width: int = 1024
    rank: int = 8
    alpha: float = 8.0
    video_lora: bool = False
    sam_lora: bool = False


@chz.chz
class DataConfig:
    root: str = "/work/data/navi"
    prepared: str = "/work/data/view_coverage"
    frames: int = 16
    image_size: int = 384
    seed: int = 42
    train_objects: int = 24
    val_objects: int = 6
    test_objects: int = 6
    clips_per_object: int = 2
    pilot_train_objects: int = 2
    pilot_val_objects: int = 2


@chz.chz
class TrainingConfig:
    objective: str = "canonical_coarse_shape"
    project: str = "atdl-video-to-3d"
    entity: str = "hprjdk"
    output: str = "/work/results/view_coverage"
    group: str = ""
    microbatch: int = 1
    accumulation: int = 4
    steps: int = 25
    learning_rate: float = 0.0001
    weight_decay: float = 0.0
    warmup: int = 5
    clip_grad: float = 1.0
    validate_every: int = 5
    eval_every: int = 25
    checkpoint_every: int = 5
    inference_steps: int = 25
    eval_examples: int = 2
    max_seconds: int = 1080
    seed: int = 42


@chz.chz
class ProvenanceConfig:
    git_head: str = ""
    git_worktree_sha256: str | None = None


@chz.chz
class ExperimentConfig:
    model: ModelConfig = chz.field(default_factory=ModelConfig)
    data: DataConfig = chz.field(default_factory=DataConfig)
    training: TrainingConfig = chz.field(default_factory=TrainingConfig)
    provenance: ProvenanceConfig = chz.field(default_factory=ProvenanceConfig)

    @property
    def variant(self) -> str:
        return {(False, False): "A", (True, False): "B", (False, True): "C", (True, True): "D"}[(self.model.video_lora, self.model.sam_lora)]

    @property
    def run_id(self) -> str:
        return f"{self.training.group}-{self.variant}"


def capture_provenance(root: Path) -> ProvenanceConfig:
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(["git", "-C", str(root), "status", "--porcelain"], text=True)
    if status.strip():
        raise RuntimeError("Commit run-relevant source before launching this sweep")
    return ProvenanceConfig(git_head=head)


def sweep_group(head: str) -> str:
    return f"view-coverage-g{head[:12]}-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
