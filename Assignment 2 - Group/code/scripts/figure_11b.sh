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
  echo 'Python environment not found; set PYTHON_BIN.' >&2
  exit 1
fi

if [[ -z "${WIDTHS:-}" ]]; then
  echo 'WIDTHS must be set' >&2
  exit 1
fi

if [[ -z "${SAMPLES:-}" ]]; then
  echo 'SAMPLES must be set' >&2
  exit 1
fi

arguments=(
  task=translation
  data_dir=/work/ATDL/data/iwslt14
  output_dir=/work/ATDL/runs/translation-figure11b
  results_dir=/work/ATDL/our-results-folder
  embedding_dims="$WIDTHS"
  sample_counts="$SAMPLES"
  max_steps=80000
  warmup_steps=4000
  eval_every=4000
  checkpoint_every=8000
  seed=0
  device=cuda
)

exec "$python_bin" "$script_dir/../launch.py" submit \
  "$script_dir/figure_11b.sh" \
  24 \
  "${arguments[@]}" \
  "$@"
