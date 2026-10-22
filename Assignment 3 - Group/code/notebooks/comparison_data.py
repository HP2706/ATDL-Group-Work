from __future__ import annotations

import json
from pathlib import Path
from typing import Any
import wandb

NEW_GROUP = "mask-pooler-g32f534db2662-2026-10-22-14-07-23"
OLD_GROUP = "view-coverage-gd1cee366ecd2-2026-10-22-11-33-45"
PROJECT = "hprjdk/atdl-video-to-3d"
REGIMES = [("mean_B", "Earlier averaging B", OLD_GROUP, "B"), ("pool_A", "Pooler A: connector", NEW_GROUP, "A"), ("pool_B", "Pooler B: video LoRA", NEW_GROUP, "B"), ("pool_C", "Pooler C: SAM LoRA", NEW_GROUP, "C"), ("pool_D", "Pooler D: both LoRAs", NEW_GROUP, "D")]


def download_comparison(output: Path) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "comparison.json"
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    api = wandb.Api(overrides={"base_url": "https://api.forge.coreweave.com"}, timeout=30)
    run_id = NEW_GROUP + "-A"
    artifacts = [a for a in api.run(PROJECT + "/" + run_id).logged_artifacts() if a.type == "reconstruction"]
    videos = sorted([a for a in artifacts if a.metadata["mode"] == "video"], key=lambda a: a.metadata["example_id"])
    bundle: dict[str, Any] = {"new_group": NEW_GROUP, "earlier_group": OLD_GROUP, "earlier_variant": "B", "objects": []}

    def download(artifact: Any, filename: str, directory: Path) -> str:
        destination = output / directory / filename
        if not destination.exists():
            artifact.get_entry(filename).download(root=str(output / directory))
        assert destination.is_file() and destination.stat().st_size > 0
        return str(destination.relative_to(output))

    for video in videos:
        record = video.metadata
        identity = record["example_id"]
        row: dict[str, Any] = {"example_id": identity, "object_id": record["object_id"], "split": record["split"], "models": []}
        assert row["split"] == "val"
        for name in ["input.mp4", "reference.png", "reference_mask.png", "ground_truth.glb"]:
            row[name] = download(video, name, Path(identity) / "inputs")
        native = api.artifact(f"{PROJECT}/{run_id}-{identity}-native_image-s25:latest")
        row["models"].append({"key": "native", "label": "Original SAM 3D (reference frame)", "mesh": download(native, "prediction.glb", Path(identity) / "native"), "metrics": native.metadata["metrics"], "run_url": f"https://forge.coreweave.com/wandb/{PROJECT}/runs/{run_id}"})
        for key, label, group, variant in REGIMES:
            name = f"{group}-{variant}"
            artifact = api.artifact(f"{PROJECT}/{name}-{identity}-video-s25:latest")
            assert artifact.metadata["object_id"] == row["object_id"] and artifact.metadata["example_id"] == identity
            row["models"].append({"key": key, "label": label, "mesh": download(artifact, "prediction.glb", Path(identity) / key), "metrics": artifact.metadata["metrics"], "run_url": f"https://forge.coreweave.com/wandb/{PROJECT}/runs/{name}"})
        bundle["objects"].append(row)
        print("Downloaded matched meshes and video:", identity, flush=True)
    manifest_path.write_text(json.dumps(bundle, indent=2))
    return bundle
