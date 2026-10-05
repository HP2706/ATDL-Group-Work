#!/usr/bin/env bash
set -euo pipefail

# Ablation X1 (see ABLATIONS.md): continue copies of the completed 400-epoch
# cifar10_resnet.sh runs at 20% noise to 1,200 epochs. Relative paths are on the work drive.

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
  architecture=resnet
  widths=3,8,12,16,64
  noise_rates=0.2
  batch_size=128
  runs_per_gpu=4
  variants.adam.epochs=1200
  variants.adam.measure_every_epochs=10
  variants.adam.schedule=constant
  variants.adam.learning_rate=0.0001
  augmentation=true
  weight_decay=0.0
  seed=0
  device=cuda
  wandb_project=atdl-double-descent
  wandb_entity=hprjdk
  extend_from=runs/sweeps/resnet-cifar10-adam-400-2026-09-26-21-52-00
  results_dir=our-results-folder/ablations/x1-horizon
)

exec "$python_bin" "$script_dir/../launch.py" submit "$script_dir/x1_horizon_resnet.sh" 8 "${arguments[@]}" "$@"
