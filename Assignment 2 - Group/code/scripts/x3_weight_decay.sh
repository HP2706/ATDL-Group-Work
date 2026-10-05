#!/usr/bin/env bash
set -euo pipefail

# Ablation X3 (see ABLATIONS.md): the figure_11a.sh augmented CNN at 20% noise with
# weight decay 5e-4, at 12,500 and all 50,000 training examples.

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
  noise_rates=0.2
  sample_counts=12500,50000
  batch_size=128
  runs_per_gpu=4
  variants.sgd.max_steps=50000
  variants.sgd.measure_every_steps=1250
  variants.sgd.schedule=inverse_sqrt
  variants.sgd.learning_rate=0.1
  variants.sgd.momentum=0.0
  augmentation=true
  weight_decay=0.0005
  seed=0
  device=cuda
  wandb_project=atdl-double-descent
  wandb_entity=hprjdk
  results_dir=our-results-folder/ablations/x3-weight-decay
)

exec "$python_bin" "$script_dir/../launch.py" submit "$script_dir/x3_weight_decay.sh" 10 "${arguments[@]}" "$@"
