"""Verify both appended-stream variants with real checkpoints on a remote GPU."""
from __future__ import annotations

from datetime import datetime
import gc
import json
from pathlib import Path
import subprocess

import fire
import torch

from video3d.config import apply_process_env, config_dict, read_config, replace_config
from video3d.evaluate import evaluate
from video3d.models.adapter import build_adapter, count_parameters
from video3d.storage import atomic_json
from video3d.train import train


def smoke(base_config: str, evaluation_config: str, output: str) -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("The appender smoke test requires a remote CUDA GPU")
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    source = subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip()
    base, evaluation = read_config(base_config), read_config(evaluation_config)
    stamp = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')
    summary: dict[str, object] = {'source_commit': source, 'variants': {}}
    for sam_lora in [False, True]:
        name = 'append-sam-lora' if sam_lora else 'append-frozen'
        cfg = replace_config(base, **{
            'condition': name, 'run_name': f'{name}-smoke-{stamp}',
            'provenance.git_head': source, 'adapter.strategy': 'append_video_mask',
            'adapter.append_queries': 1370, 'sam3d.lora': sam_lora,
            'encoder.vjepa.video_lora': False, 'train.batch_size': 1,
            'train.effective_batch_size': 2, 'train.epochs': 0, 'train.max_steps': 2,
            'train.ckpt_every': 1, 'train.log_every': 1, 'train.max_seconds': 1200,
            'data.num_workers': 0, 'paths.output_root': str(root/'results'),
            'paths.scratch_root': str(root/'scratch'),
            'val.loss_every': 1000, 'val.sample_every': 1000,
            'val.heavy_epochs': [], 'val.final_full_validation': False,
            'val.epoch_media_samples': 0})
        atomic_json(root/'configs'/f'{name}.json', config_dict(cfg))
        apply_process_env(cfg)
        parameters = count_parameters(build_adapter(cfg.adapter))
        if train(cfg, stop_at=1)['step'] != 1 or train(cfg)['step'] != 2:
            raise AssertionError('Strict checkpoint resume did not complete two updates')
        run_dir = root/'results'/cfg.run_name
        state = torch.load(run_dir/'ckpt_2.pt', weights_only=False, map_location='cpu')
        if not all(torch.isfinite(value).all() for value in state['adapter'].values()):
            raise AssertionError('Non-finite adapter state after training')
        if bool([key for key in state['extra'] if key.startswith('sam.')]) != sam_lora:
            raise AssertionError('Saved SAM LoRA state differs from requested setting')
        if any(key.startswith('encoder.') for key in state['extra']):
            raise AssertionError('V-JEPA must remain frozen')
        eval_cfg = replace_config(cfg, **{
            'data.source': evaluation.data.source,
            'data.eval_split': evaluation.data.eval_split,
            'paths.data_root': evaluation.paths.data_root,
            'paths.cache_root': evaluation.paths.cache_root,
            'sampler.mode': evaluation.sampler.mode, 'sampler.steps': evaluation.sampler.steps,
            'eval.k': evaluation.eval.k})
        result = evaluate(eval_cfg, str(run_dir), limit=1)
        rows = [json.loads(line) for line in result.read_text().splitlines()]
        if len(rows) != 1 or 'chamfer' not in rows[0]:
            raise AssertionError('Expected one evaluated reconstruction')
        summary['variants'][name] = {'steps': 2, 'adapter_parameters': parameters,
                                    'evaluation': str(result), 'sam_lora': sam_lora}
        atomic_json(root/'summary.json', summary)
        del state
        gc.collect()
        torch.cuda.empty_cache()
    atomic_json(root/'ready.json', summary)


if __name__ == '__main__':
    fire.Fire(smoke)
