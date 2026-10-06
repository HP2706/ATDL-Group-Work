#!/usr/bin/env bash

set -euo pipefail

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
  task=translation
  data_dir=/work/ATDL/data/iwslt14
  output_dir=/work/ATDL/runs/translation-figure3-8k
  results_dir=/work/ATDL/our-results-folder

  embedding_dims=144,152,160,168,176,184,192,200
  sample_counts=4000,18000

  max_steps=8000
  warmup_steps=4000
  eval_every=1000
  checkpoint_every=1000

  seed=0
  device=cuda
)

exec "$python_bin" "$script_dir/../launch.py" submit \
  "$script_dir/figure_3_8k_high.sh" \
  12 \
  "${arguments[@]}" \
  "$@"
