#!/usr/bin/env bash
set -Eeuo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$CODE_ROOT"
uv sync --locked --group generation --group gpu_runtime --group dev
uv run --no-sync python scripts/install_models.py --metrics=True --runtime=True
