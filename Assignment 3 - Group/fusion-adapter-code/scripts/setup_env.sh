#!/usr/bin/env bash
# Build the GPU environment once: conda env with torch 2.7.1+cu128 and the compiled extensions
# SAM 3D imports (pytorch3d, kaolin, spconv, gsplat), built for the local GPU. Runs on A100/H100
# (sm_80/sm_90) and B200 (sm_100). SAM 3D's own pinned stack (torch 2.5.1+cu121) has no sm_100
# kernels, so we do not use it. SAM 3D's source is never changed.
#
#   bash scripts/fetch_upstream.sh --repos-only   # pinned SAM 3D and V-JEPA 2 checkouts first
#   bash scripts/setup_env.sh                     # then this (also the UCloud Initialization script)
#   source scripts/activate.sh                    # in every job
#
# Everything lives under $PROJECT_DIR (default /work/vidsam3d). Finished steps leave a stamp in
# $ENV_DIR/.setup/, so a rerun continues where it stopped and is quick once all are done.
# FORCE=1 redoes all steps. SKIP_GPU_CHECK=1 builds without a GPU (Docker image; then set
# TORCH_CUDA_ARCH_LIST, default "8.0;9.0;10.0").
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/vidsam3d}"
ENV_DIR="${ENV_DIR:-$PROJECT_DIR/envs/vidsam3d}"
CONDA_ROOT="${CONDA_ROOT:-$PROJECT_DIR/miniforge3}"
CODE_DIR="${CODE_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SAM3D_REPO="${SAM3D_REPO:-$PROJECT_DIR/sam-3d-objects}"
SKIP_GPU_CHECK="${SKIP_GPU_CHECK:-0}"

PYTHON_VERSION=3.11
CUDA_VERSION=12.8                 # toolkit for building extensions; matches the torch wheels
TORCH_VERSION=2.7.1
TORCHVISION_VERSION=0.22.1
TORCH_INDEX=https://download.pytorch.org/whl/cu128
KAOLIN_VERSION=0.18.0
SPCONV_PACKAGE="${SPCONV_PACKAGE:-spconv-cu126}"             # CUDA-12 build; only imported
PYTORCH3D_COMMIT=75ebeeaea0908c5527e7b1e305fbc7681382db47   # SAM 3D's pin
GSPLAT_COMMIT=2323de5905d5e90e035f792fe65bad0fedd413e7      # SAM 3D's pin

export CONDA_PKGS_DIRS="${CONDA_PKGS_DIRS:-$PROJECT_DIR/conda/pkgs}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-$PROJECT_DIR/pip-cache}"
export TORCH_HOME="${TORCH_HOME:-$PROJECT_DIR/torch_home}"
export MAX_JOBS="${MAX_JOBS:-16}"
mkdir -p "$CONDA_PKGS_DIRS" "$PIP_CACHE_DIR" "$TORCH_HOME"
unset PIP_EXTRA_INDEX_URL PIP_FIND_LINKS

log() { printf '\n==> %s\n' "$*"; }
die() { printf 'ERROR: %s\n' "$*" >&2; exit 1; }

# step <name> <function>: run once, then remember it in $ENV_DIR/.setup/<name>.done
step() {
    local stamp="$ENV_DIR/.setup/$1.done"
    if [[ -f "$stamp" && "${FORCE:-0}" != 1 ]]; then
        log "$1: done"
        return
    fi
    log "$1"
    "$2"
    mkdir -p "$ENV_DIR/.setup"
    touch "$stamp"
}

# ---- preconditions ----
if [[ "$SKIP_GPU_CHECK" != 1 ]]; then
    command -v nvidia-smi >/dev/null || die "no nvidia-smi: run this in a GPU job (or SKIP_GPU_CHECK=1)"
    nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
    # build the extensions for the GPU in this machine, e.g. 10.0 on B200
    TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader | sort -u | paste -sd ';')}"
fi
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0;10.0}"
[[ -f "$SAM3D_REPO/patching/hydra" ]] || \
    die "no SAM 3D checkout at $SAM3D_REPO; run: bash scripts/fetch_upstream.sh --repos-only"
getent hosts pypi.org download.pytorch.org github.com >/dev/null || \
    die "cannot resolve pypi.org / download.pytorch.org / github.com (network or DNS)"
if [[ ! -f "$CONDA_ROOT/etc/profile.d/conda.sh" ]]; then
    log "installing Miniforge into $CONDA_ROOT"
    curl -fL -o "$PROJECT_DIR/Miniforge3.sh" \
        https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
    bash "$PROJECT_DIR/Miniforge3.sh" -b -p "$CONDA_ROOT"
fi
# shellcheck source=/dev/null
source "$CONDA_ROOT/etc/profile.d/conda.sh"

# ---- steps ----
create_env() {
    # nvcc + CUDA headers + gcc 13 for building extensions; libgl/libegl for open3d (libGL.so.1)
    conda create -y -p "$ENV_DIR" -c conda-forge \
        "python=$PYTHON_VERSION" git "cuda-version=$CUDA_VERSION" cuda-nvcc cuda-cudart-dev \
        cuda-cccl libcublas-dev libcusparse-dev libcusolver-dev libcurand-dev \
        "gcc_linux-64=13" "gxx_linux-64=13" libgl libegl libglib
}

install_torch() {
    python -m pip install "torch==$TORCH_VERSION" "torchvision==$TORCHVISION_VERSION" \
        --index-url "$TORCH_INDEX"
}

check_torch_gpu() {
    python - <<'EOF'
import torch
cap = torch.cuda.get_device_capability(0)
archs = torch.cuda.get_arch_list()
print(f"torch {torch.__version__}, CUDA {torch.version.cuda}, GPU sm_{cap[0]}{cap[1]}, built for {archs}")
assert f"sm_{cap[0]}{cap[1]}" in archs, "this torch build has no kernels for the GPU"
x = torch.randn(1024, 1024, device="cuda")
torch.cuda.synchronize()
print("GPU matmul ok:", tuple((x @ x).shape))
EOF
}

install_python_deps() {
    python -m pip install -r "$CODE_DIR/requirements.gpu.txt" -r "$CODE_DIR/requirements.txt"
    # INSTALL_PACKAGE=0 (Docker): the image installs vidsam3d after copying the code
    [[ "${INSTALL_PACKAGE:-1}" == 0 ]] || python -m pip install --no-deps -e "$CODE_DIR"
}

install_spconv() {
    python -m pip install "$SPCONV_PACKAGE==2.3.8"
}

install_gsplat() {
    # imported by SAM 3D's renderers but never run in Stage 1: skip its CUDA build (it JITs)
    BUILD_NO_CUDA=1 python -m pip install --no-build-isolation \
        "gsplat @ git+https://github.com/nerfstudio-project/gsplat.git@$GSPLAT_COMMIT"
}

install_kaolin() {
    # NVIDIA's wheel for this torch/CUDA pair if it exists, else a source build
    local index="https://nvidia-kaolin.s3.us-east-2.amazonaws.com/torch-${TORCH_VERSION}_cu128.html"
    if ! python -m pip install "kaolin==$KAOLIN_VERSION" -f "$index"; then
        log "no kaolin wheel for torch $TORCH_VERSION+cu128; building v$KAOLIN_VERSION from source"
        IGNORE_TORCH_VER=1 FORCE_CUDA=1 python -m pip install --no-build-isolation \
            "kaolin @ git+https://github.com/NVIDIAGameWorks/kaolin.git@v$KAOLIN_VERSION"
    fi
}

install_pytorch3d() {
    # source build against the installed torch for $TORCH_CUDA_ARCH_LIST (10-30 min per arch)
    FORCE_CUDA=1 python -m pip install --no-build-isolation \
        "pytorch3d @ git+https://github.com/facebookresearch/pytorch3d.git@$PYTORCH3D_COMMIT"
}

patch_hydra() {
    # SAM 3D's own post-install step (hydra PR 2863); rewrites a file inside the env's hydra
    (cd "$SAM3D_REPO" && python ./patching/hydra)
}

verify() {
    SAM3D_REPO="$SAM3D_REPO" GPU="$([[ "$SKIP_GPU_CHECK" == 1 ]] && echo 0 || echo 1)" python - <<'EOF'
import importlib, os, sys
os.environ.setdefault("LIDRA_SKIP_INIT", "1")
sys.path.insert(0, os.environ["SAM3D_REPO"])
failed = []
for name in ["torch", "pytorch3d", "pytorch3d._C", "kaolin", "spconv.pytorch", "gsplat", "open3d",
             "moge", "utils3d", "sam3d_objects.pipeline.inference_pipeline_pointmap"]:
    try:
        mod = importlib.import_module(name)
        print(f"[OK  ] {name} {getattr(mod, '__version__', '')}")
    except Exception as e:
        failed.append(name)
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e).splitlines()[0][:160]}")
if os.environ["GPU"] == "1" and not failed:
    import torch
    from pytorch3d.ops import knn_points
    print("[OK  ] pytorch3d knn on GPU:", tuple(knn_points(torch.rand(1, 8, 3, device="cuda"),
                                                          torch.rand(1, 9, 3, device="cuda")).dists.shape))
sys.exit(1 if failed else 0)
EOF
}

step create_env create_env
set +u                                      # conda's activation hooks use unset variables
conda activate "$ENV_DIR"
set -u
export CUDA_HOME="$CONDA_PREFIX"
export CPATH="$CONDA_PREFIX/targets/x86_64-linux/include:${CPATH:-}"   # conda-forge CUDA headers
export LIBRARY_PATH="$CONDA_PREFIX/targets/x86_64-linux/lib:$CONDA_PREFIX/lib:${LIBRARY_PATH:-}"
export LD_LIBRARY_PATH="$CONDA_PREFIX/lib:${LD_LIBRARY_PATH:-}"
export CC="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc" CXX="$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++"

step install_torch install_torch
[[ "$SKIP_GPU_CHECK" == 1 ]] || check_torch_gpu     # fail fast if torch cannot use this GPU
step install_python_deps install_python_deps
step install_spconv install_spconv
step install_gsplat install_gsplat
step install_kaolin install_kaolin
step install_pytorch3d install_pytorch3d
step patch_hydra patch_hydra
log "verify"
verify
log "done (built for TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST). In every job: source $CODE_DIR/scripts/activate.sh"
