"""W&B mirrors persisted training records and complete evaluation tables."""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import chz

from video3d.config import ExperimentConfig
from video3d.storage import atomic_json


EVALUATION_METADATA = {'schema_version', 'k', 'r', 'clip_len', 'num_objects',
                       'shuffle_frames', 'wrong_video', 'frame_ids'}


def numeric_metrics(values: dict[str, Any], prefix: str = '') -> dict[str, float | int | bool]:
    result: dict[str, float | int | bool] = {}
    for key, value in values.items():
        if not prefix and key in EVALUATION_METADATA:
            continue
        name = f'{prefix}/{key}' if prefix else key
        if isinstance(value, dict):
            result.update(numeric_metrics(value, name))
        elif isinstance(value, (float, int)) and math.isfinite(value):
            result[name] = value
    return result


class RunLogger:
    def __init__(self, cfg: ExperimentConfig) -> None:
        self.run: Any = None
        self.name = cfg.run_name
        if cfg.logging.wandb:
            import wandb
            self.run = wandb.init(project=cfg.logging.project, entity=cfg.logging.entity,
                                  name=cfg.run_name, id=cfg.run_name, resume='allow',
                                  config=chz.asdict(cfg))
            self.run.define_metric('eval/sample_index')
            self.run.define_metric('eval/*', step_metric='eval/sample_index')

    def log(self, values: dict[str, Any], step: int | None = None) -> None:
        if self.run is not None:
            self.run.log(values, step=step)

    def log_evaluation(self, record: dict[str, Any], sample_index: int) -> None:
        self.log({'eval/sample_index':sample_index,
                  **{f'eval/{key}':value for key,value in numeric_metrics(record).items()}})

    def finish_evaluation(self, records: list[dict[str, Any]], directory: Path,
                          label: str, expected_count: int) -> dict[str, Any]:
        metrics = [numeric_metrics(row) for row in records]
        keys = sorted({key for row in metrics for key in row})
        summary: dict[str, Any] = {'eval/objects':len(records), 'eval/expected_objects':expected_count,
                                  'eval/complete':len(records) == expected_count}
        for key in keys:
            values = [row[key] for row in metrics if key in row]
            summary[f'eval/mean/{key}'] = sum(values)/len(values)
            summary[f'eval/count/{key}'] = len(values)
        atomic_json(directory/f'{label}.summary.json', summary)
        if self.run is not None:
            import wandb
            self.run.summary.update(summary)
            self.run.log({'eval/results':wandb.Table(
                columns=['sample_id', *keys],
                data=[[row['sample_id'], *[values.get(key) for key in keys]]
                      for row,values in zip(records,metrics,strict=True)])})
            artifact = wandb.Artifact(f'{label}-evaluation', type='evaluation', metadata=summary)
            for suffix in ['jsonl', 'meta.json', 'summary.json']:
                path = directory/f'{label}.{suffix}'
                if path.exists():
                    artifact.add_file(str(path), name=path.name)
            self.run.log_artifact(artifact)
        return summary

    def finish(self) -> None:
        if self.run is not None:
            self.run.finish()
