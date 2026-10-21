from __future__ import annotations

import gc
import time
from pathlib import Path
import chz
import wandb
from contextlib import ExitStack
from datetime import datetime
import torch
from torch.utils.data import default_collate
from .config import ExperimentConfig, capture_provenance
from .data import Clips, read_manifest
from .model import NativeSAM
from .storage import atomic_json


def profile(config: ExperimentConfig) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("Profiling requires the remote GPU")
    config = chz.replace(config, provenance=capture_provenance(Path(__file__).resolve().parents[4]))
    model_config = chz.replace(config.model, video_lora=True, sam_lora=True)
    rows = [row for row in read_manifest(Path(config.data.prepared) / "manifest.jsonl") if row.split == "train"]
    dataset = Clips(rows, config.data)
    results: list[dict[str, float | int | str]] = []
    capacity = torch.cuda.get_device_properties(0).total_memory
    model = NativeSAM(model_config, config.data.frames, torch.device("cuda"))
    parameters = [p for p in model.parameters() if p.requires_grad]
    counts = {"total": sum(p.numel() for p in model.parameters()), "trainable": sum(p.numel() for p in parameters), "connector": sum(p.numel() for p in model.conditioning.projector.parameters()), "lora_targets": model.lora_targets}
    run_id = (config.training.group or ("profile-" + datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))) + "-profile"
    with ExitStack() as cleanup:
        run = wandb.init(project=config.training.project, entity=config.training.entity, group=config.training.group or run_id, id=run_id, name=run_id, job_type="profile", config=chz.asdict(config), resume="allow")
        cleanup.enter_context(run)
        for size in [1, 2, 4]:
            optimizer = torch.optim.AdamW(parameters, lr=config.training.learning_rate)
            batch = default_collate([dataset[i % len(dataset)] for i in range(size)])
            video, reference = batch["video"].cuda(), batch["reference"].cuda()
            target = {key: value.cuda() for key, value in batch["target"].items()}
            torch.cuda.reset_peak_memory_stats()
            start = time.monotonic()
            try:
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        loss, _ = model.loss(video, reference, target)
                    loss.backward()
                    optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.monotonic() - start
                results.append({"microbatch": size, "seconds_per_step": elapsed / 2, "sequences_per_second": 2 * size / elapsed, "peak_allocated_bytes": torch.cuda.max_memory_allocated(), "peak_reserved_bytes": torch.cuda.max_memory_reserved(), "status": "passed"})
            except torch.cuda.OutOfMemoryError:
                results.append({"microbatch": size, "status": "oom"})
            finally:
                optimizer.zero_grad(set_to_none=True)
                del optimizer, video, reference, target, batch
                gc.collect()
                torch.cuda.empty_cache()
        safe = [row for row in results if row["status"] == "passed" and row["peak_reserved_bytes"] <= capacity - 2 * 1024**3]
        if not safe:
            raise RuntimeError("No profiled batch preserves 2 GiB of evaluation headroom; do not start sweep")
        chosen = max(safe, key=lambda row: row["sequences_per_second"])
        atomic_json(Path(config.training.output) / "profile.json", {"schema_version": 2, "group": config.training.group, "git_head": config.provenance.git_head, "device": torch.cuda.get_device_name(), "capacity_bytes": capacity, "parameters": counts, "measurements": results, "selected_microbatch": chosen["microbatch"], "selected_accumulation": 4 // int(chosen["microbatch"]), "effective_batch": 4})
        run.log({"profile/batches": wandb.Table(data=[[row.get(key) for key in ["microbatch", "status", "seconds_per_step", "sequences_per_second", "peak_reserved_bytes"]] for row in results], columns=["microbatch", "status", "seconds_per_step", "sequences_per_second", "peak_reserved_bytes"])})
        artifact = wandb.Artifact(run_id + "-measurements", type="profile")
        artifact.add_file(str(Path(config.training.output) / "profile.json"))
        run.log_artifact(artifact)
