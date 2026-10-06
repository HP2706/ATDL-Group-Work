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

if [[ -z "${WIDTHS:-}" ]]; then
  echo 'WIDTHS must be set' >&2
  exit 1
fi

arguments=(
  task=translation
  data_dir=/work/ATDL/data/iwslt14

  # Deliberately point to the existing 40k run directories.
  output_dir=/work/ATDL/runs/translation-figure3-40k
  results_dir=/work/ATDL/our-results-folder

  embedding_dims="$WIDTHS"
  sample_counts=4000,18000

  max_steps=80000
  warmup_steps=4000
  eval_every=4000
  checkpoint_every=4000

  seed=0
  device=cuda
  resume_existing=true
)

exec "$python_bin" "$script_dir/../launch.py" submit \
  "$script_dir/figure_3_80k_resume.sh" \
  24 \
  "${arguments[@]}" \
  "$@"

