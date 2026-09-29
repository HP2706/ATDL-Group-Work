"""Short A100 runs for estimating the four CIFAR training families."""

import json
from datetime import datetime
from pathlib import Path
import sys
from typing import Any

import modal

sys.path.insert(0, str(Path(__file__).resolve().parent / "code"))
from modal_vision_runtime import image, volume

WIDTHS = (1, 30, 64)
NOISE_RATES = (0.0, 0.1, 0.2)
ARCHITECTURES = ("resnet", "cnn")
DATASETS = ("cifar10", "cifar100")

app = modal.App("atdl-double-descent-vision-pilot")


@app.function(
    image=image,
    gpu="A100",
    cpu=4,
    memory=16384,
    timeout=3600,
    volumes={"/root/shared": volume},
)
def pilot(run_name: str, max_new_conditions: int) -> list[dict[str, Any]]:
    import sys
    import time

    import torch
    from tqdm import tqdm

    sys.path.insert(0, "/root/code")
    from vision_train import VisionConfig, train

    torch.set_num_threads(4)
    shared = Path("/root/shared")
    summary_path = shared / "runs" / run_name / "pilot_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summaries = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    gpu_name = torch.cuda.get_device_name(0)
    total = len(ARCHITECTURES) * len(DATASETS) * len(WIDTHS) * len(NOISE_RATES)
    priority = ("resnet", "cifar100", 64, 0.0)
    conditions = [priority] + [
        (architecture, dataset, width, noise)
        for architecture in ARCHITECTURES
        for dataset in DATASETS
        for width in WIDTHS
        for noise in NOISE_RATES
        if (architecture, dataset, width, noise) != priority
    ]
    completed_now = 0
    overall = tqdm(
        total=total, initial=len(summaries), desc="A100 vision pilot", unit="run",
        bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} runs [{elapsed}]",
        position=0, dynamic_ncols=True,
    )
    for architecture, dataset, width, noise in conditions:
        if completed_now >= max_new_conditions:
            break
        condition = f"{architecture}-{dataset}-width{width}-noise{round(noise * 100):02d}"
        if condition in summaries:
            continue
        output_dir = shared / "runs" / run_name / condition
        optimizer = "adam" if architecture == "resnet" else "sgd"
        config = VisionConfig(
            architecture=architecture,
            dataset=dataset,
            data_dir=str(shared / "data" / "cifar"),
            output_dir=str(output_dir),
            results_dir=str(shared / "results" / run_name),
            width=width,
            label_noise=noise,
            augmentation=True,
            optimizer=optimizer,
            schedule="constant" if optimizer == "adam" else "inverse_sqrt",
            learning_rate=0.0001 if optimizer == "adam" else 0.1,
            epochs=2,
            batch_size=128,
            eval_every=3910,
            checkpoint_every=391,
            seed=0,
            device="cuda",
            download=True,
        )
        existing = sorted(path for path in output_dir.glob("*/") if (path / "config.json").is_file())
        if len(existing) > 1:
            raise ValueError(f"multiple run directories for {condition}: {existing}")
        resume = str(existing[0]) if existing else None
        print(f"Pilot {len(summaries) + 1}/{total}: {condition}", flush=True)
        started = time.perf_counter()
        run_dir = train(config, resume=resume, progress_position=1, checkpoint_callback=volume.commit)
        wall_seconds = time.perf_counter() - started
        timings = [json.loads(line) for line in (run_dir / "epoch_timings.jsonl").read_text().splitlines()]
        full_epochs = [entry for entry in timings if entry["full_epoch"]]
        if not full_epochs:
            raise ValueError(f"no complete epoch timing for {condition}")
        summary = {
            "condition": condition,
            "architecture": architecture,
            "dataset": dataset,
            "width": width,
            "label_noise": noise,
            "optimizer": optimizer,
            "gpu": gpu_name,
            "epochs": 2,
            "seconds_per_epoch": full_epochs[-1]["seconds"],
            "updates_per_epoch": full_epochs[-1]["updates"],
            "wall_seconds": wall_seconds,
            "run_dir": str(run_dir),
        }
        summaries[condition] = summary
        temporary = summary_path.with_name(summary_path.name + ".tmp")
        temporary.write_text(json.dumps(summaries, indent=2) + "\n")
        temporary.replace(summary_path)
        volume.commit()
        completed_now += 1
        overall.update(1)
        print(json.dumps(summary), flush=True)
    overall.close()
    return list(summaries.values())


@app.local_entrypoint()
def main(run_name: str = "", max_new_conditions: int = 36) -> None:
    selected_name = run_name or f"vision-pilot-{datetime.now().strftime('%Y-%m-%d-%H-%M-%S')}"
    print(f"Pilot run: {selected_name}", flush=True)
    with modal.enable_output():
        summaries = pilot.remote(selected_name, max_new_conditions)
    print(f"Completed {len(summaries)} conditions; summary: runs/{selected_name}/pilot_summary.json", flush=True)
