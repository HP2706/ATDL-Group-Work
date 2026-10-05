#!/usr/bin/env bash
set -euo pipefail

# Ablation X4 (see ABLATIONS.md): the cifar10_resnet.sh 20% setting with (a) Adam at LR 1e-3
# and (b) SGD + momentum 0.9 with dynamic drop from LR 0.01 (paper Figures 16 and 18).

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
  widths=12,64
  noise_rates=0.2
  batch_size=128
  runs_per_gpu=4
  variants.adam.epochs=400
  variants.adam.measure_every_epochs=10
  variants.adam.schedule=constant
  variants.adam.learning_rate=0.001
  variants.sgd.epochs=400
  variants.sgd.measure_every_epochs=10
  variants.sgd.schedule=dynamic_drop
  variants.sgd.learning_rate=0.01
  variants.sgd.momentum=0.9
  augmentation=true
  weight_decay=0.0
  seed=0
  device=cuda
  wandb_project=atdl-double-descent
  wandb_entity=hprjdk
  results_dir=our-results-folder/ablations/x4-optimizer
)

exec "$python_bin" "$script_dir/../launch.py" submit "$script_dir/x4_optimizer.sh" 8 "${arguments[@]}" "$@"
