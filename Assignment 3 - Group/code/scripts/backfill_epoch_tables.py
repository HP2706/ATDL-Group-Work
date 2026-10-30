"""Combine existing W&B epoch tables without downloading or duplicating media."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import fire
import wandb
from wandb.sdk.artifacts.artifact_manifest_entry import ArtifactManifestEntry


def combine_tables(tables: list[dict[str, Any]]) -> dict[str, Any]:
    if not tables:
        raise ValueError("No epoch reconstruction tables found")
    columns = tables[0]["columns"]
    rows: dict[tuple[int, str, int], list[Any]] = {}
    for table in tables:
        if table["columns"] != columns:
            raise ValueError("Epoch table schemas differ")
        for row in table["data"]:
            key = (int(row[columns.index("epoch")]), str(row[columns.index("clip")]),
                   int(row[columns.index("object")]))
            if key in rows and rows[key] != row:
                raise ValueError(f"Conflicting reconstruction for {key}")
            rows[key] = row
    return {**tables[0], "data": [rows[key] for key in sorted(rows)],
            "nrows": len(rows), "ncols": len(columns), "log_mode": "IMMUTABLE"}


def media_paths(table: dict[str, Any]) -> set[str]:
    return {cell["path"] for row in table["data"] for cell in row
            if isinstance(cell, dict) and "path" in cell}


def backfill(run_id: str, entity: str = "hprjdk", project: str = "atdl-video-to-3d",
             directory: str = "/tmp/sam3dvid-wandb-history") -> dict[str, Any]:
    api = wandb.Api(timeout=60)
    source = api.run(f"{entity}/{project}/{run_id}")
    output = Path(directory) / run_id
    backup = output / "original-summary.json"
    retry = source.state == "failed" and backup.exists() and source.summary.get("epoch") == 10
    if source.state != "finished" and not retry:
        raise ValueError(f"Run must be finished before backfill: {source.state}")
    original_summary = dict(source.summary._json_dict)
    output.mkdir(parents=True, exist_ok=True)
    if not backup.exists():
        backup.write_text(json.dumps(original_summary, indent=2))
    tables: list[dict[str, Any]] = []
    references: dict[str, tuple[str, str, int]] = {}
    for artifact in source.logged_artifacts():
        name = "epoch/reconstructions.table.json"
        if artifact.type != "run_table" or name not in artifact.manifest.entries:
            continue
        table_path = artifact.get_entry(name).download(root=str(output / artifact.version))
        table = json.loads(Path(table_path).read_text())
        tables.append(table)
        for path in media_paths(table):
            entry = artifact.get_entry(path)
            if path in references and references[path][1] != entry.digest:
                raise ValueError(f"Conflicting media path: {path}")
            references[path] = (entry.ref_url(), entry.digest, (len(tables) - 1) // 5)
    merged = combine_tables(tables)
    epoch_column = merged["columns"].index("epoch")
    epochs = sorted({row[epoch_column] for row in merged["data"]})
    if epochs != list(range(1, 11)) or merged["nrows"] != 100:
        raise ValueError(f"Expected 10 epochs / 100 rows, got {epochs} / {merged['nrows']}")
    table_file = output / "reconstruction_history.table.json"
    table_file.write_text(json.dumps(merged, allow_nan=False))
    combined = wandb.Artifact(f"{run_id}-reconstruction-history", type="reconstruction_history",
                              metadata={"epochs": epochs, "rows": merged["nrows"]})
    combined.add_file(str(table_file), name="epoch/reconstructions.table.json")
    with wandb.init(entity=entity, project=project, id=run_id, resume="must",
                    dir=str(output)) as resumed:
        # The service limits each artifact to five direct dependencies. Keep
        # explicit two-level references instead of flattening ten epoch artifacts.
        for group in sorted({value[2] for value in references.values()}):
            part = wandb.Artifact(f"{run_id}-history-media-{group}",
                                  type="reconstruction_history_media")
            paths = [path for path, value in references.items() if value[2] == group]
            for path in sorted(paths):
                reference, digest, _ = references[path]
                part.manifest.add_entry(ArtifactManifestEntry(
                    path=path, digest=digest, size=0, ref=reference))
            published = resumed.log_artifact(part)
            published.wait()
            for path in sorted(paths):
                entry = published.get_entry(path)
                combined.manifest.add_entry(ArtifactManifestEntry(
                    path=path, digest=entry.digest, size=0, ref=entry.ref_url()))
        saved = resumed.log_artifact(combined)
        saved.wait()
        resumed.save(str(table_file), base_path=str(output), policy="now")
        reference = saved.get_entry("epoch/reconstructions.table.json").ref_url()
        descriptor = {"_type": "table-file", "path": table_file.name,
                      "artifact_path": reference, "_latest_artifact_path": reference,
                      "sha256": hashlib.sha256(table_file.read_bytes()).hexdigest(),
                      "size": table_file.stat().st_size, "ncols": merged["ncols"],
                      "nrows": merged["nrows"], "log_mode": "IMMUTABLE"}
        resumed.summary["epoch/reconstructions"] = descriptor
        resumed.summary["epoch/reconstructions_all_epochs"] = descriptor
        resumed.summary["epoch/reconstruction_history_epochs"] = epochs
        resumed.summary["epoch/reconstruction_history_rows"] = merged["nrows"]
    verified = wandb.Api(timeout=60).run(f"{entity}/{project}/{run_id}")
    descriptor = verified.summary["epoch/reconstructions"]
    if descriptor["nrows"] != 100 or verified.state != "finished":
        raise ValueError("Updated summary was not persisted correctly")
    for key in ("loss", "step", "epoch", "epoch/loss", "grad_norm"):
        if verified.summary.get(key) != original_summary.get(key):
            raise ValueError(f"Training summary changed unexpectedly: {key}")
    result = {"run": run_id, "rows": 100, "epochs": epochs,
              "media_references": len(references), "artifact": saved.qualified_name,
              "url": verified.url, "state": verified.state}
    (output / "verification.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result), flush=True)
    return result


if __name__ == "__main__":
    fire.Fire(backfill)
