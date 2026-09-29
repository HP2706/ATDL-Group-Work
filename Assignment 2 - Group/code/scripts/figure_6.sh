#!/usr/bin/env bash
set -euo pipefail

# Reduced Figure 6: clean CIFAR-10 CNNs without augmentation, SGD versus Adam.
# Run `bash code/scripts/figure_6.sh plan=true` to print both grids locally.
# Run `bash code/scripts/figure_6.sh` with the configured launch backend.

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
  widths=1,2,3,4,6,8,12,16,24,32,64
  noise_rates=0.0
  batch_size=128
  runs_per_gpu=4
  #variants.sgd.schedule=inverse_sqrt
  #variants.sgd.learning_rate=0.1
  #variants.sgd.momentum=0.0
  #variants.sgd.max_steps=50000
  #variants.sgd.measure_every_steps=1250
  variants.adam.schedule=constant
  variants.adam.learning_rate=0.0001
  variants.adam.epochs=400
  variants.adam.measure_every_epochs=10
  augmentation=false
  weight_decay=0.0
  seed=0
  device=cuda
  wandb_project=atdl-double-descent
  wandb_entity=hprjdk
)

exec "$python_bin" "$script_dir/../launch.py" submit "$script_dir/figure_6.sh" 10 "${arguments[@]}" "$@"
