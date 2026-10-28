#!/usr/bin/env bash
# UCloud "Batch processing" script:  scripts/ucloud/run.sh <command> [args...]
# The "Initialization script" is scripts/setup_env.sh (builds the env once, then only verifies).
# Only /work persists; outputs land in /work/vidsam3d/outputs (PROJECT_DIR=/work/vidsam3d).
set -euo pipefail
export VIDSAM3D_ENV=ucloud
exec "$(dirname "${BASH_SOURCE[0]}")/../run.sh" "$@"
