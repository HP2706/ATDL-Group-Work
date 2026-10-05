#!/usr/bin/env bash
set -euo pipefail

# Ablation X1 (see ABLATIONS.md): continue copies of the completed 50,000-step
# augmented CNN runs at 20% noise to 200,000 steps. Relative paths are on the work drive.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
repo_dir="$(cd "$project_dir/.." && pwd)"
python_bin="${PYTHON_BIN:-$repo_dir/.venv/bin/python}"
if [[ ! -x "$python_bin" ]]; then
  python_bin="${PYTHON_BIN:-$project_dir/.venv/bin/python}"
fi
if [[ ! -x "$python_bin" ]]; then
  echo 'Python environment not found; set PYTHON_BIN to the project virtual environment.' >&2
  exit 1
fi

arguments=(
  dataset=cifar10
  architecture=cnn
  widths=16,32,48,64,128
  noise_rates=0.2
  batch_size=128
  runs_per_gpu=4
  variants.sgd.max_steps=200000
  variants.sgd.measure_every_steps=1250
  variants.sgd.schedule=inverse_sqrt
  variants.sgd.learning_rate=0.1
  variants.sgd.momentum=0.0
  augmentation=true
  weight_decay=0.0
  seed=0
  device=cuda
  wandb_project=atdl-double-descent
  wandb_entity=hprjdk
  extend_from=runs/sweeps/cnn-cifar10-sgd-50000-2026-09-27-10-37-31
  results_dir=our-results-folder/ablations/x1-horizon
)

exec "$python_bin" "$script_dir/../launch.py" submit "$script_dir/x1_horizon_cnn.sh" 10 "${arguments[@]}" "$@"
