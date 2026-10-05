#!/usr/bin/env bash
set -euo pipefail

# Ablation X2 (see ABLATIONS.md): new seeds (initialization, data order, and noise mask)
# for the cifar10_resnet.sh 20% grid. Seed 0 is the completed sweep; submit again with seed=2.

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
  widths=4,8,12,16,24,64
  noise_rates=0.2
  batch_size=128
  runs_per_gpu=4
  variants.adam.epochs=400
  variants.adam.measure_every_epochs=10
  variants.adam.schedule=constant
  variants.adam.learning_rate=0.0001
  augmentation=true
  weight_decay=0.0
  seed=1
  device=cuda
  wandb_project=atdl-double-descent
  wandb_entity=hprjdk
  results_dir=our-results-folder/ablations/x2-seeds
)

exec "$python_bin" "$script_dir/../launch.py" submit "$script_dir/x2_seeds.sh" 8 "${arguments[@]}" "$@"
