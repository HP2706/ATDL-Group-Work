#!/usr/bin/env bash
set -euo pipefail

# Non-augmented CIFAR-10 CNN Adam sweep; includes the clean Figure 6 arm.

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
  noise_rates=0.0,0.1,0.2
  batch_size=128
  runs_per_gpu=4
  variants.adam.epochs=400
  variants.adam.measure_every_epochs=10
  variants.adam.schedule=constant
  variants.adam.learning_rate=0.0001
  augmentation=false
  weight_decay=0.0
  seed=0
  device=cuda
  wandb_project=atdl-double-descent
  wandb_entity=hprjdk
)

exec "$python_bin" "$script_dir/../launch.py" submit "$script_dir/cifar10_cnn_adam_no_data_aug.sh" 10 "${arguments[@]}" "$@"
