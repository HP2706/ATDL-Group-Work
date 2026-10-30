"""Bounded batch profiling through the existing trainer, before finalizing configs."""
from __future__ import annotations

import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import fire

from video3d.config import config_dict, read_config, replace_config
from video3d.storage import atomic_json


def profile(root: str, batches: tuple[int, ...] = (8, 16, 32), steps: int = 5,
            resume: bool = False, attempt: str = "", only: str = "") -> None:
    folder = Path(root)
    matrix = json.loads((folder/'matrix.json').read_text())
    if only and only not in {row['name'] for row in matrix['entries'] if not row['baseline']}:
        raise ValueError(f'Unknown trainable variant: {only}')
    progress = folder/'profiles'/f'{only}-progress.json' if only else folder/'profiles/progress.json'
    reports: dict[str, Any] = json.loads(progress.read_text()) if resume and progress.exists() else {}
    suffix = f'-{attempt}' if attempt else '' 
    for entry in matrix['entries']:
        if entry['baseline'] or (only and entry['name'] != only):
            continue
        cfg = read_config(entry['config'])
        if entry['name'] in reports:
            if reports[entry['name']]['selected']['batch_size'] != cfg.train.batch_size:
                raise ValueError('Completed profile differs from resolved batch size')
            continue
        candidates: list[dict[str, Any]] = []
        for batch in batches:
            candidate = replace_config(cfg, **{
                'run_name':f'{cfg.run_name}-profile-b{batch}{suffix}', 'train.batch_size':batch,
                'train.benchmark_steps':steps, 'train.max_seconds':300,
                'logging.wandb':False, 'data.num_workers':2})
            path = folder/'profiles'/f'{entry["name"]}-b{batch}{suffix}.json'
            atomic_json(path, config_dict(candidate))
            log = path.with_suffix('.log')
            with log.open('w') as output:
                result = subprocess.run([sys.executable, '-c',
                    'import sys;from video3d.config import read_config,apply_process_env;'
                    'from video3d.train import train;'
                    'c=read_config(sys.argv[1]);apply_process_env(c);train(c,benchmark=True)',
                    str(path)], stdout=output, stderr=subprocess.STDOUT, timeout=420)
            if result.returncode:
                if 'out of memory' in log.read_text().lower():
                    candidates.append({'batch_size':batch, 'status':'oom', 'log':str(log)})
                    break
                raise RuntimeError(f'Profiling failed: {log}')
            report_path = Path(cfg.paths.output_root)/(candidate.run_name+'_benchmark')/'benchmark.json'
            report = json.loads(report_path.read_text())
            candidates.append({**report, 'status':'ok', 'videos_per_second':
                               cfg.train.effective_batch_size/report['s_per_step']})
        successful = [candidate for candidate in candidates if candidate['status'] == 'ok']
        if not successful:
            raise RuntimeError(f'No batch size fits: {entry["name"]}; see {folder}/profiles')
        selected = max(successful, key=lambda row: row['videos_per_second'])
        updated = replace_config(cfg, **{'train.batch_size':selected['batch_size']})
        atomic_json(Path(entry['config']), config_dict(updated))
        evaluation = replace_config(read_config(entry['evaluation']), **{'train.batch_size':selected['batch_size']})
        atomic_json(Path(entry['evaluation']), config_dict(evaluation))
        reports[entry['name']] = {'candidates':candidates, 'selected':selected,
            'accumulation_steps':cfg.train.effective_batch_size//selected['batch_size'],
            'stopped_at':'OOM or effective-batch ceiling 32'}
        atomic_json(progress, reports)
    ready = folder/'profiles'/f'{only}-ready.json' if only else folder/'profiles-ready.json'
    atomic_json(ready, reports)


if __name__ == '__main__':
    fire.Fire(profile)
