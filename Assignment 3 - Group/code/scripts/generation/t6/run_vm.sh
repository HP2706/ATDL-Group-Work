#!/usr/bin/env bash
set -Eeuo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
source "$CODE_ROOT/.venv/bin/activate"
exec python "$CODE_ROOT/scripts/generation/t6/run_sam3d.py" "$@"
