# shellcheck shell=bash
# Activate the environment built by setup_env.sh. Use: source scripts/activate.sh
# Sets the persistent cache paths and the variables the adapter code reads.

PROJECT_DIR="${PROJECT_DIR:-/work/vidsam3d}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/envs/vidsam3d}"
CONDA_ROOT="${CONDA_ROOT:-$PROJECT_DIR/miniforge3}"

# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"
conda activate "$ENV_DIR"

export PROJECT_DIR
export CUDA_HOME="$CONDA_PREFIX"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"   # libGL for open3d
export HF_HOME="${HF_HOME:-$PROJECT_DIR/hf_home}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/torch_home}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_DIR/pip-cache}"
export SAM3D_REPO="${SAM3D_REPO:-$PROJECT_DIR/sam-3d-objects}"
export VJEPA2_REPO="${VJEPA2_REPO:-$PROJECT_DIR/vjepa2}"
export VIDSAM3D_ENV="${VIDSAM3D_ENV:-ucloud}"
