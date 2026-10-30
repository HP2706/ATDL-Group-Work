#!/usr/bin/env bash
set -euo pipefail
task_root="${1:?evaluation root required}"
code_root="$task_root/repo/Assignment 3 - Group/code"
mkdir -p "$task_root/logs"
exec > >(tee -a "$task_root/logs/preparation.log") 2>&1
finish() {
  status=$?
  printf '{"exit_code":%d,"finished_utc":"%s"}\n' "$status" "$(date -u +%FT%TZ)" > "$task_root/logs/preparation-exit.json"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
export PYTHONPATH="$code_root/src"
sudo apt-get update -qq
sudo apt-get install -y -qq libgl1 libegl1 libgl1-mesa-dri libglu1-mesa libxrender1 > "$task_root/logs/preparation-system.log" 2>&1
python_binary='/work/MathiasElmegaardNielsen#6181/omni-gso-data/runtime/cpu-venv/bin/python'
cd "$code_root"
TASK_ROOT="$task_root" "$python_binary" - <<'PY'
from concurrent.futures import ProcessPoolExecutor
import json
import os
from pathlib import Path
import time
import numpy as np
from video3d.data.scan_prepare import write_manifest
from video3d.data.scanned import ScanRecord, audit
from video3d.data.generation.assets import asset_id_for
from video3d.baselines.analysis import cache_visibility
root = Path(os.environ['TASK_ROOT'])
source = Path('/work/MathiasElmegaardNielsen#6181/omni-gso-data/data/gso')
data = root/'data'
data.mkdir(exist_ok=True)
if not (data/'assets').exists():
    (data/'assets').symlink_to(source/'assets', target_is_directory=True)
selected = json.loads((root/'selection.json').read_text())['objects']
if len(selected) != 250 or len(set(selected)) != 250:
    raise ValueError('Expected exactly 250 unique selected objects')
pending = {name: asset_id_for(name) for name in selected}
complete = []
while pending:
    ready = [name for name, identity in pending.items() if (source/'assets'/identity/'record.json').is_file()]
    tasks = []
    batch = []
    for name in ready:
        row = ScanRecord.model_validate_json((source/'assets'/pending[name]/'record.json').read_text())
        ids = np.linspace(0, len(row.frames)-1, 4).round().astype(int).tolist()
        tasks.append((str(data/row.mesh), str(data/row.cameras), ids,
                      str(data/'visibility'/(row.object_id+'.npz'))))
        batch.append(row)
    if tasks:
        with ProcessPoolExecutor(max_workers=2) as pool:
            for result in pool.map(cache_visibility, *zip(*tasks, strict=True)):
                print('Visibility cache complete', result['reference_visible_fraction'], flush=True)
        complete.extend(batch)
        for name in ready:
            del pending[name]
        write_manifest(data, complete)
        print(json.dumps({'ready': len(complete), 'waiting_for_renderer': len(pending)}), flush=True)
    if pending:
        time.sleep(10)
audit(str(data))
(root/'preparation-complete.json').write_text(json.dumps({'objects':250, 'frames':8000,
    'evaluation_frames':4, 'visibility_samples':8192, 'selection':'selection.json',
    'render_source':str(source), 'rendering_performed_here':False}))
PY
"$python_binary" -m pytest tests/test_baselines.py -q > "$task_root/logs/preparation-tests.log" 2>&1
