#!/usr/bin/env bash
set -Eeuo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ "${ACTIVATE:-1}" == 1 ]]; then
  source "$CODE_ROOT/.venv/bin/activate"
fi
if [[ "${STAGE_TO_SCRATCH:-0}" == 1 ]]; then
  : "${DATA_ROOT:?Set DATA_ROOT to the existing export}"
  : "${SCRATCH_ROOT:?Set SCRATCH_ROOT}"
  split=train
  [[ "${1:-}" == eval ]] && split=eval
  args=("$@")
  for ((i=0;i<${#args[@]};i++)); do
    if [[ "${args[$i]}" == --split && $((i+1)) -lt ${#args[@]} ]]; then
      split="${args[$((i+1))]}"
    fi
  done
  [[ "$split" == val ]] && split=train
  stage="$SCRATCH_ROOT/dataset"
  mkdir -p "$stage"
  echo "staging $split export to $stage"
  cp -a "$DATA_ROOT/$split" "$stage/"
  export DATA_ROOT="$stage"
fi
export LIDRA_SKIP_INIT=1 ATTN_BACKEND=sdpa
exec python -m video3d "$@"
