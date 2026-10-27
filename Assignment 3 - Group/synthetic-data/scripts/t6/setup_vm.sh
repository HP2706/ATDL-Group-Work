#!/usr/bin/env bash
# Install SAM 3D Objects and its checkpoints on a fresh GPU VM (Ubuntu, NVIDIA driver present).
# Follows sam-3d-objects/doc/setup.md at the pinned commit. Safe to re-run: finished steps are skipped.
#
# Usage: bash scripts/t6/setup_vm.sh            (asks for a HuggingFace token once)
set -euo pipefail

SAM3D_COMMIT=f91db411c50efee93d8db7aeb323885650f6f722
SAM3D_DIR=${SAM3D_DIR:-$HOME/sam-3d-objects}
CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv

# open3d and the renderers need OpenGL/EGL libraries that headless VM images leave out.
sudo apt-get update -qq
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
    libgl1 libegl1 libglib2.0-0 libsm6 libxrender1 libxext6 libgomp1 >/dev/null

if [ ! -x "$CONDA_DIR/bin/mamba" ]; then
    curl -fsSL -o /tmp/miniforge.sh \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash /tmp/miniforge.sh -b -p "$CONDA_DIR"
fi
source "$CONDA_DIR/etc/profile.d/conda.sh"

if [ ! -d "$SAM3D_DIR/.git" ]; then
    git clone https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_DIR"
fi
cd "$SAM3D_DIR"
git fetch --depth 1 origin "$SAM3D_COMMIT" 2>/dev/null || true
git checkout -q "$SAM3D_COMMIT"

if ! conda env list | grep -q '^sam3d-objects '; then
    "$CONDA_DIR/bin/mamba" env create -y -f environments/default.yml
fi
# conda's activation scripts read unset variables.
set +u; conda activate sam3d-objects; set -u

if ! python -c "import sam3d_objects, pytorch3d, kaolin" 2>/dev/null; then
    # setup.md also lists pypi.ngc.nvidia.com, which no longer resolves; pip then retries every
    # package. The nvidia-pyindex dependency writes it into pip.conf files, so those are ignored too.
    export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu121"
    export PIP_CONFIG_FILE=/dev/null
    pip install -e '.[dev]'
    # pytorch3d's dependency on torch is broken upstream; installing it as a second step works around it.
    pip install -e '.[p3d]'
    export PIP_FIND_LINKS="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-2.5.1_cu121.html"
    pip install -e '.[inference]'
    ./patching/hydra
fi
python -c "import pytorch3d, torch; from pytorch3d import _C; assert torch.cuda.is_available(); print('torch', torch.__version__, 'cuda ok')"

if [ ! -f checkpoints/hf/pipeline.yaml ]; then
    pip install 'huggingface-hub[cli]<1.0'
    hf auth whoami >/dev/null 2>&1 || hf auth login
    hf download --repo-type model --local-dir checkpoints/hf-download --max-workers 1 facebook/sam-3d-objects
    mv checkpoints/hf-download/checkpoints checkpoints/hf
    rm -rf checkpoints/hf-download
fi

echo "SAM 3D ready in $SAM3D_DIR (conda env sam3d-objects)"
