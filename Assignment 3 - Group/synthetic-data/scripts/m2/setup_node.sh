#!/usr/bin/env bash
# Prepare a GPU node for rendering: uv environment and Blender 4.5 LTS. Run from the repo root.
# Safe to re-run.
set -euo pipefail

BLENDER_VERSION=4.5.15
BLENDER_DIR=.blender/blender-${BLENDER_VERSION}-linux-x64

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
ldd --version | head -1

if ! command -v uv >/dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    source "$HOME/.local/bin/env"
fi
uv sync

if [ ! -x "$BLENDER_DIR/blender" ]; then
    mkdir -p .blender
    curl -fsSL "https://download.blender.org/release/Blender${BLENDER_VERSION%.*}/blender-${BLENDER_VERSION}-linux-x64.tar.xz" \
        | tar xJ -C .blender
fi
"$BLENDER_DIR/blender" --version | head -1
echo "Node ready. Next: bash scripts/m2/smoke_test.sh"
