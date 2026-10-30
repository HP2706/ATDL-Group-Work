#!/usr/bin/env bash
# Install a one-line wrapper as UCloud's foreground batchScript for each variant.
set -Eeuo pipefail
root="${1:?resolved experiment root}"
variant="${2:?variant name}"
python="${3:?prepared runtime Python}"
code_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PYTHONPATH="$code_root/src"
mkdir -p "$root/logs"
exec > >(tee -a "$root/logs/$variant.log") 2>&1
exec 9>"$root/$variant.lock"
flock -n 9
finish() {
  result=$?
  printf '{"exit_code":%d,"finished_utc":"%s"}\n' "$result" "$(date -u +%FT%TZ)" > "$root/logs/$variant-exit.json"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
cd "$code_root"
config="$root/configs/$variant.json"
evaluation="$root/configs/$variant-eval.json"
test -f "$root/matrix.json"
if [[ "$variant" == prepare-targets ]]; then
  EXPERIMENT_ROOT="$root" "$python" - <<'PYTHON'
import os
from pathlib import Path
from video3d.config import apply_process_env, read_config
from video3d.data.targets import cache_latents
from video3d.models.sam3d import build_sam3d
from video3d.storage import atomic_json
from video3d.utils import pick_device
root = Path(os.environ['EXPERIMENT_ROOT'])
cfg = read_config(root/'configs/fusion-frozen.json')
apply_process_env(cfg)
device = pick_device(cfg.device)
model = build_sam3d(cfg, device)
splits = ['train'] if cfg.data.source == 'synthetic' else ['train','val']
reports = [cache_latents(cfg, split, model, device, batch_size=32) for split in splits]
evaluation = read_config(root/'configs/sam3d-baseline-eval.json')
reports.append(cache_latents(evaluation, 'eval', model, device, batch_size=32))
atomic_json(root/'targets-ready.json', {'reports':reports})
PYTHON
elif [[ "$variant" == sam3d-baseline ]]; then
  "$python" -m video3d eval-json "$evaluation" zero-shot
else
  test -f "$root/targets-ready.json"
  "$python" -m video3d train-json "$config"
  run_dir="$root/results/$("$python" -c 'import sys;from video3d.config import apply_process_env, read_config;print(read_config(sys.argv[1]).run_name)' "$config")"
  RUN_CONFIG="$config" RUN_DIRECTORY="$run_dir" EXPERIMENT_ROOT="$root" "$python" - <<'PYTHON'
import json
import os
from pathlib import Path
from video3d.config import read_config
from video3d.storage import atomic_json
cfg = read_config(os.environ['RUN_CONFIG'])
run = Path(os.environ['RUN_DIRECTORY'])
matrix = json.loads((Path(os.environ['EXPERIMENT_ROOT'])/'matrix.json').read_text())
expected = matrix['steps_per_epoch']*cfg.train.epochs if cfg.train.epochs else cfg.train.max_steps
latest = json.loads((run/'latest.json').read_text())
if latest['step'] != expected:
    raise RuntimeError(f"Incomplete training: {latest['step']}/{expected} updates. Resume the checkpoint before evaluation.")
atomic_json(run/'training-complete.json', {'steps':latest['step'], 'expected_steps':expected, 'epochs':cfg.train.epochs})
PYTHON
  "$python" -m video3d eval-json "$evaluation" "$run_dir"
fi
