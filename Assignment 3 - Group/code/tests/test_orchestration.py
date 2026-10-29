from __future__ import annotations
import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from video3d.config import load_config


def test_sweep_uses_one_trainer_and_full_configs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import video3d.sweep as module
    monkeypatch.setattr(module.sys, 'platform', 'linux')
    calls: list[list[str]] = []
    def execute(command: list[str], **kwargs: object) -> SimpleNamespace:
        calls.append(command)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(module.subprocess, 'run', execute)
    cfg=load_config('pooler', overrides=[f'paths.output_root={tmp_path}', 'run_name=matrix',
                                       'train.max_seconds=1'])
    directory=module.sweep(cfg,max_seconds=600)
    assert len(calls)==8
    assert sum('train-json' in command for command in calls)==4
    assert sum('eval-json' in command for command in calls)==4
    resolved=[json.loads(path.read_text()) for path in directory.glob('*.json') if path.name!='sweep.json']
    assert len(resolved)==4
    assert {(c['encoder']['vjepa']['video_lora'],c['sam3d']['lora']) for c in resolved}=={(False,False),(True,False),(False,True),(True,True)}
    assert all(c['adapter']['strategy']=='pooler' for c in resolved)


def test_preserve_generated_output(tmp_path: Path) -> None:
    from video3d.data.generation.clipio import preserve_existing
    target=tmp_path/'clip'
    target.mkdir()
    (target/'rgb.png').write_bytes(b'original')
    preserve_existing(target)
    assert not target.exists()
    preserved=list(tmp_path.glob('clip.preserved-*'))
    assert len(preserved)==1
    assert (preserved[0]/'rgb.png').read_bytes()==b'original'


def test_visibility_metrics_are_canonical(tmp_path: Path) -> None:
    import numpy as np
    import torch
    from video3d.metrics import visibility_metrics
    vox=torch.zeros(64,64,64,dtype=torch.bool)
    vox[32,32,32]=True
    path=tmp_path/'visibility.npz'
    np.savez(path,points=np.array([[0.5/64]*3]),reference_visible=[True],newly_revealed=[False],never_observed=[False])
    result=visibility_metrics(vox,str(path))
    assert result['canonical/reference_visible/recall_002']==1.0
    assert result['canonical/reference_visible/completeness_distance']==0.0
