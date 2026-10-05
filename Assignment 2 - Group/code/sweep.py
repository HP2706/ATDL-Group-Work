"""Schedule a CIFAR vision sweep on a Linux GPU host."""

import hashlib
import math
import os
import platform
import selectors
import shutil
import signal
import subprocess
import sys
import tarfile
import urllib.request
from collections import deque
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from types import FrameType

import chz
from tqdm import tqdm


CODE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CODE_DIR.parent
CIFAR_URL = "https://data.brainchip.com/dataset-mirror/cifar10/cifar-10-python.tar.gz"
CIFAR_MD5 = "c58f30108f718f92721af3b95e74349a"
Condition = tuple[str, int, float, int | None]


@chz.chz(typecheck=True)
class OptimizerConfig:
    schedule: str
    learning_rate: float
    epochs: int | None = None
    max_steps: int | None = None
    measure_every_epochs: int | None = None
    measure_every_steps: int | None = None
    momentum: float = 0.0

    @chz.validate
    def validate(self) -> None:
        if (self.epochs is None) == (self.max_steps is None):
            raise ValueError("Choose exactly one training horizon: epochs or max_steps")
        if self.epochs is not None:
            if self.epochs <= 0 or self.measure_every_epochs is None or self.measure_every_steps is not None:
                raise ValueError("An epoch sweep needs positive epochs and measure_every_epochs only")
        if self.max_steps is not None:
            if self.max_steps <= 0 or self.measure_every_steps is None or self.measure_every_epochs is not None:
                raise ValueError("A step sweep needs positive max_steps and measure_every_steps only")
        if self.measure_every_epochs is not None and self.measure_every_epochs <= 0:
            raise ValueError("measure_every_epochs must be positive")
        if self.measure_every_steps is not None and self.measure_every_steps <= 0:
            raise ValueError("measure_every_steps must be positive")
        if self.learning_rate <= 0 or not 0 <= self.momentum < 1:
            raise ValueError("Learning rate must be positive and momentum in [0, 1)")


@chz.chz(typecheck=True)
class SweepConfig:
    dataset: str
    architecture: str
    widths: tuple[int, ...]
    noise_rates: tuple[float, ...]
    batch_size: int
    runs_per_gpu: int
    variants: dict[str, OptimizerConfig]
    augmentation: bool
    weight_decay: float
    seed: int
    device: str
    wandb_project: str
    wandb_entity: str
    sample_count: int | None = None
    sample_counts: tuple[int, ...] = ()
    data_dir: str = ""
    results_dir: str = ""
    run_root: str = ""
    extend_from: str = ""
    plan: bool = False

    @chz.validate
    def validate(self) -> None:
        if self.dataset not in {"cifar10", "cifar100"}:
            raise ValueError(f"Unsupported dataset: {self.dataset}")
        if self.architecture not in {"resnet", "cnn"}:
            raise ValueError(f"Unsupported architecture: {self.architecture}")
        if not self.variants or set(self.variants) - {"adam", "sgd"}:
            raise ValueError("Variants must contain adam, sgd, or both")
        if not self.widths or not self.noise_rates:
            raise ValueError("The sweep needs at least one width and noise rate")
        if len(set(self.widths)) != len(self.widths) or len(set(self.noise_rates)) != len(self.noise_rates):
            raise ValueError("The sweep contains duplicate conditions")
        if any(width <= 0 for width in self.widths) or any(not 0 <= noise <= 1 for noise in self.noise_rates):
            raise ValueError("Widths must be positive and noise rates must lie between zero and one")
        if self.batch_size <= 0 or self.runs_per_gpu <= 0:
            raise ValueError("Batch size and runs per GPU must be positive")
        if self.sample_count is not None and self.sample_count <= 0:
            raise ValueError("sample_count must be positive")
        if self.sample_count is not None and self.sample_counts:
            raise ValueError("Use sample_count or sample_counts, not both")
        if len(set(self.sample_counts)) != len(self.sample_counts) or any(count <= 0 for count in self.sample_counts):
            raise ValueError("sample_counts must contain distinct positive sizes")
        if self.weight_decay < 0:
            raise ValueError("Weight decay must be nonnegative")
        if "adam" in self.variants and self.variants["adam"].momentum:
            raise ValueError("Momentum is only valid for SGD")

    @chz.init_property
    def sweep_root(self) -> Path:
        if self.run_root:
            return Path(self.run_root)
        timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
        if len(self.variants) == 1:
            optimizer, variant = next(iter(self.variants.items()))
            horizon = f"{variant.epochs}-epochs" if variant.epochs is not None else f"{variant.max_steps}-steps"
            name = f"{self.architecture}-{self.dataset}-{optimizer}-{horizon}"
        else:
            name = f"{self.architecture}-{self.dataset}-optimizers"
        return CODE_DIR / "runs" / "sweeps" / f"{name}-{timestamp}"

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir) if self.data_dir else PROJECT_DIR / "data"

    @property
    def results_path(self) -> Path:
        return Path(self.results_dir) if self.results_dir else PROJECT_DIR / "our-results-folder"

    def condition_directory(self, optimizer: str, width: int, noise: float, sample_count: int | None) -> Path:
        percent = format(Decimal(str(noise)) * 100, "f")
        if "." in percent:
            percent = percent.rstrip("0").rstrip(".")
        root = self.sweep_root if len(self.variants) == 1 else self.sweep_root / optimizer
        if self.sample_counts:
            assert sample_count is not None
            root = root / f"samples-{sample_count}"
        return root / f"noise-{percent.replace('.', 'p').zfill(2)}" / f"width-{width}"


class Job:
    def __init__(self, condition: Condition, gpu: int, process: subprocess.Popen[bytes], descriptor: int) -> None:
        self.condition = condition
        self.gpu = gpu
        self.process = process
        self.descriptor = descriptor
        self.buffer = b""
        self.last_step = 0


def conditions(config: SweepConfig) -> list[Condition]:
    counts = config.sample_counts if config.sample_counts else (config.sample_count,)
    return [(optimizer, width, noise, sample_count) for sample_count in counts
            for noise in config.noise_rates for width in config.widths for optimizer in config.variants]


def existing_run(directory: Path) -> Path | None:
    runs = sorted(path.parent for path in directory.glob("*/config.json"))
    if len(runs) > 1:
        raise ValueError(f"Multiple runs found in {directory}: {runs}")
    return runs[0] if runs else None


def validate_existing(config: SweepConfig, optimizer: str, width: int, noise: float,
                      sample_count: int | None, train_size: int) -> Path | None:
    from vision_train import VisionConfig

    previous = existing_run(config.condition_directory(optimizer, width, noise, sample_count))
    if previous is None:
        return None
    saved = VisionConfig.model_validate_json((previous / "config.json").read_text())
    if saved != VisionConfig.from_sweep(config, optimizer, width, noise, sample_count, train_size):
        raise ValueError(f"Saved configuration differs from this sweep: {previous}")
    if not (previous / "completed.json").exists() and not (previous / "checkpoint.pt").exists():
        raise ValueError(f"Incomplete run has no checkpoint: {previous}")
    return previous


def extend_run(config: SweepConfig, condition: Condition, train_size: int) -> None:
    """Copy a checkpointed run from extend_from; only its paths and a longer horizon may change."""
    from src.training.common import save_config
    from vision_train import VisionConfig

    target = config.condition_directory(*condition)
    if existing_run(target) is not None:
        return
    source = existing_run(Path(config.extend_from) / target.relative_to(config.sweep_root))
    if source is None or not (source / "checkpoint.pt").exists():
        raise ValueError(f"No checkpointed run to extend for {condition} in {config.extend_from}")
    saved = VisionConfig.model_validate_json((source / "config.json").read_text())
    extended = VisionConfig.from_sweep(config, *condition, train_size)
    movable = {"data_dir", "output_dir", "results_dir", "epochs", "max_steps"}
    old, new = saved.model_dump(exclude=movable), extended.model_dump(exclude=movable)
    changed = [name for name in old if old[name] != new[name]]
    if changed:
        raise ValueError(f"Cannot extend {source}; settings differ: {', '.join(changed)}")
    if (saved.epochs is None) != (extended.epochs is None) or \
            (saved.epochs or saved.max_steps) >= (extended.epochs or extended.max_steps):
        raise ValueError(f"Cannot extend {source}; the new horizon must be longer in the same unit")
    shutil.copytree(source, target / source.name, ignore=shutil.ignore_patterns("completed.json", "test_predictions.pt"))
    save_config(target / source.name, extended)


def verify_md5(path: Path) -> None:
    digest = hashlib.md5()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != CIFAR_MD5:
        raise ValueError(f"CIFAR archive checksum failed: {path}")


def prepare_cifar(dataset: str, data_dir: Path) -> int:
    from torchvision.datasets import CIFAR10, CIFAR100

    data_dir.mkdir(parents=True, exist_ok=True)
    if dataset == "cifar100":
        train_data = CIFAR100(root=data_dir, train=True, download=True)
        CIFAR100(root=data_dir, train=False, download=False)
        return len(train_data)
    if dataset != "cifar10":
        raise ValueError(f"Unsupported CIFAR dataset: {dataset}")

    archive = data_dir / "cifar-10-python-brainchip.tar.gz"
    if not (data_dir / "cifar-10-batches-py").is_dir():
        if not archive.exists():
            partial = archive.with_name(archive.name + ".part")
            offset = partial.stat().st_size if partial.exists() else 0
            request = urllib.request.Request(CIFAR_URL, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urllib.request.urlopen(request, timeout=60) as response:
                if offset and response.status != 206:
                    raise ValueError(f"Download server cannot resume {partial}; partial file was preserved")
                with partial.open("ab") as output:
                    for chunk in iter(lambda: response.read(1024 * 1024), b""):
                        output.write(chunk)
            verify_md5(partial)
            partial.replace(archive)
        verify_md5(archive)
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(data_dir, filter="data")
    train_data = CIFAR10(root=data_dir, train=True, download=False)
    CIFAR10(root=data_dir, train=False, download=False)
    return len(train_data)


def visible_gpus() -> list[int]:
    if platform.system() != "Linux":
        raise RuntimeError("Training must run on a remote Linux GPU host")
    output = subprocess.run(["nvidia-smi", "--list-gpus"], capture_output=True, text=True, check=True).stdout
    gpu_count = len(output.splitlines())
    if gpu_count == 0:
        raise RuntimeError("No NVIDIA GPU is visible")
    selected = os.environ.get("CUDA_VISIBLE_DEVICES")
    return [int(value) for value in selected.split(",")] if selected else list(range(gpu_count))


def launch(config: SweepConfig, condition: Condition, gpu: int, previous: Path | None, train_size: int) -> Job:
    optimizer, width, noise, sample_count = condition
    directory = config.condition_directory(optimizer, width, noise, sample_count)
    directory.mkdir(parents=True, exist_ok=True)
    from vision_train import VisionConfig

    options = VisionConfig.from_sweep(config, optimizer, width, noise, sample_count, train_size).model_dump(exclude_none=True)
    read_fd, write_fd = os.pipe()
    command = [sys.executable, str(CODE_DIR / "vision_train.py")]
    command.extend(f"--{key}={value}" for key, value in options.items())
    command.append(f"--progress_fd={write_fd}")
    if previous is not None:
        command.append(f"--resume={previous}")
    environment = os.environ.copy()
    environment["CUDA_VISIBLE_DEVICES"] = str(gpu)
    with (directory / "train.log").open("a") as log:
        process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT, pass_fds=(write_fd,))
    os.close(write_fd)
    return Job(condition, gpu, process, read_fd)


def steps_per_run(config: SweepConfig, optimizer: str, sample_count: int) -> int:
    variant = config.variants[optimizer]
    if variant.max_steps is not None:
        return variant.max_steps
    assert variant.epochs is not None
    return variant.epochs * math.ceil(sample_count / config.batch_size)


def run_sweep(config: SweepConfig) -> None:
    if config.wandb_project and not os.environ.get("WANDB_API_KEY") and os.environ.get("WANDB_MODE") != "disabled":
        raise ValueError("WANDB_API_KEY must be set for W&B logging; set WANDB_MODE=disabled to train without it")
    for name, path in (("DATA_DIR", config.data_path), ("RESULTS_DIR", config.results_path), ("RUN_ROOT", config.sweep_root)):
        if not path.resolve().is_relative_to(Path("/work")):
            raise ValueError(f"{name} must be under persistent /work: {path}")

    gpu_ids = visible_gpus()
    train_size = prepare_cifar(config.dataset, config.data_path)
    if any(count is not None and count > train_size for _, _, _, count in conditions(config)):
        raise ValueError(f"A sample count exceeds the {config.dataset} training split of {train_size}")
    from vision_train import VisionConfig

    for optimizer, width, noise, sample_count in conditions(config):
        VisionConfig.from_sweep(config, optimizer, width, noise, sample_count, train_size)
    if config.extend_from:
        for condition in conditions(config):
            extend_run(config, condition, train_size)
    previous_runs = {condition: validate_existing(config, *condition, train_size) for condition in conditions(config)}

    config.sweep_root.mkdir(parents=True, exist_ok=True)
    config.results_path.mkdir(parents=True, exist_ok=True)
    pending = deque(condition for condition, previous in previous_runs.items()
                    if previous is None or not (previous / "completed.json").exists())
    skipped = [condition for condition in previous_runs if condition not in pending]
    total_steps = sum(steps_per_run(config, optimizer, count or train_size)
                      for optimizer, _, _, count in previous_runs)
    initial_steps = sum(steps_per_run(config, optimizer, count or train_size)
                        for optimizer, _, _, count in skipped)
    slots = deque(gpu for gpu in gpu_ids for _ in range(config.runs_per_gpu))
    active: dict[int, Job] = {}
    selector = selectors.DefaultSelector()
    failed: list[Condition] = []
    checkpointed: list[Condition] = []
    completed = 0
    stopping = False

    def request_stop(signum: int, frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True
        for job in list(active.values()):
            job.process.send_signal(signal.SIGTERM)

    original_handler = signal.signal(signal.SIGINT, request_stop)
    print(f"Sweep: {config.sweep_root} | GPUs: {gpu_ids} | runs per GPU: {config.runs_per_gpu}")
    bar = tqdm(total=total_steps, initial=initial_steps,
               desc="All vision runs", unit="step")
    try:
        while pending or active:
            while pending and slots and not stopping:
                gpu = slots.popleft()
                condition = pending.popleft()
                job = launch(config, condition, gpu, previous_runs[condition], train_size)
                active[job.descriptor] = job
                selector.register(job.descriptor, selectors.EVENT_READ)
            if not active:
                break
            for key, _ in selector.select():
                descriptor = int(key.fd)
                job = active[descriptor]
                optimizer, width, noise, sample_count = job.condition
                run_steps = steps_per_run(config, optimizer, sample_count or train_size)
                chunk = os.read(descriptor, 65536)
                if chunk:
                    job.buffer += chunk
                    lines = job.buffer.split(b"\n")
                    job.buffer = lines.pop()
                    for line in lines:
                        step = int(line)
                        if step < job.last_step or step > run_steps:
                            raise ValueError(f"Invalid progress from {job.condition}: {step}")
                        bar.update(step - job.last_step)
                        job.last_step = step
                    continue
                selector.unregister(descriptor)
                os.close(descriptor)
                del active[descriptor]
                slots.append(job.gpu)
                exit_code = job.process.wait()
                finished = existing_run(config.condition_directory(optimizer, width, noise, sample_count))
                if exit_code != 0:
                    failed.append(job.condition)
                    tqdm.write(f"Failed: optimizer={optimizer} width={width} noise={noise:g} samples={sample_count or train_size}; see train.log")
                elif finished is not None and (finished / "completed.json").exists():
                    bar.update(run_steps - job.last_step)
                    completed += 1
                    tqdm.write(f"Completed: optimizer={optimizer} width={width} noise={noise:g} samples={sample_count or train_size}")
                else:
                    checkpointed.append(job.condition)
                    tqdm.write(f"Checkpointed: optimizer={optimizer} width={width} noise={noise:g} samples={sample_count or train_size}")
    finally:
        signal.signal(signal.SIGINT, original_handler)
        for job in active.values():
            job.process.send_signal(signal.SIGTERM)
        for job in active.values():
            job.process.wait()
            os.close(job.descriptor)
        selector.close()
        bar.close()
    print(f"Completed: {completed}; already complete: {len(skipped)}; checkpointed: {len(checkpointed)}; failed: {len(failed)}")
    if stopping or checkpointed or failed:
        raise SystemExit(1)

def main(config: SweepConfig) -> None:
    if config.plan:
        print(f"{len(conditions(config))} {config.dataset}/{config.architecture} runs")
        if config.extend_from:
            print(f"Continues copies of the matching runs in {config.extend_from}")
        for optimizer, width, noise, sample_count in conditions(config):
            variant = config.variants[optimizer]
            horizon = f"{variant.epochs} epochs" if variant.epochs is not None else f"{variant.max_steps} steps"
            print(f"optimizer={optimizer} width={width} noise={noise:g} samples={sample_count or 'full'} horizon={horizon}")
    else:
        run_sweep(config)


if __name__ == "__main__":
    chz.nested_entrypoint(main)
