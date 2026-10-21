#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
# This file is the provider-native Batch processing parameter, not an SSH/nohup launcher.
# UCloud ends this allocation when this foreground script exits, including failure.
RUN_ROOT=/work/hp-video
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/job.lock"
flock -n 9 || { echo 'Another experiment owns this staging directory.' >&2; exit 1; }
for directory in results dependencies cache data; do
  mkdir -p "$RUN_ROOT/$directory"
  if [[ ! -e "/work/$directory" ]]; then
    ln -s "$RUN_ROOT/$directory" "/work/$directory"
  elif [[ "$(readlink -f "/work/$directory")" != "$RUN_ROOT/$directory" ]]; then
    echo "Unexpected existing /work/$directory; refusing to overwrite it." >&2
    exit 1
  fi
done
exec > >(tee -a "$RUN_ROOT/batch.log") 2>&1
finish() {
  status=$?
  printf '{"exit_code":%s,"finished_at":"%s"}\n' "$status" "$(date -u +%FT%TZ)" > "$RUN_ROOT/job-exit.json"
  sync
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
if [[ ! -f /work/hp-video/runtime.env ]]; then
  echo 'Missing private runtime.env containing HF_READ_TOKEN and WANDB_API_KEY.' >&2
  exit 1
fi
set -a
. /work/hp-video/runtime.env
set +a
: "${WANDB_API_KEY:?Required for W&B logging}"
: "${HF_READ_TOKEN:?Required for gated model access}"
: "${HP_SOURCE_COMMIT:?Pin the reviewed source commit}"
export HF_HOME=/work/cache/huggingface TORCH_HOME=/work/cache/torch
export WANDB_DIR=/work/results/wandb WANDB_CONSOLE=off WANDB_MODE=online
mkdir -p "$WANDB_DIR"
export LIDRA_SKIP_INIT=1 SAM3D_GEOMETRY_ONLY=1 ATTN_BACKEND=sdpa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
if [[ "${HP_TRAINING_APPROVED:-0}" != 1 ]]; then
  echo 'Reviewed run specification must be approved before allocating experiment work.' >&2
  exit 2
fi
# 115-minute foreground deadline leaves five minutes for sync/provider shutdown.
# timeout owns the whole process group; it does not depend on the laptop staying online.
timeout --signal=INT --kill-after=60 6900 bash -s <<'PIPELINE'
set -Eeuo pipefail
clone_at() {
  local name=$1 commit=$2 url=$3
  if [[ ! -d /work/dependencies/"$name"/.git ]]; then
    git clone --filter=blob:none --no-checkout "$url" /work/dependencies/"$name"
  fi
  git -C /work/dependencies/"$name" fetch --depth 1 origin "$commit"
  git -C /work/dependencies/"$name" checkout --detach "$commit"
}
if [[ ! -d /work/ATDL-Group-Work/.git ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/HP2706/ATDL-Group-Work.git /work/ATDL-Group-Work
fi
git -C /work/ATDL-Group-Work fetch origin hp
git -C /work/ATDL-Group-Work checkout --detach "$HP_SOURCE_COMMIT"
CODE='/work/ATDL-Group-Work/Assignment 3 - Group/code'
clone_at sam-3d-objects f91db411c50efee93d8db7aeb323885650f6f722 https://github.com/facebookresearch/sam-3d-objects.git
clone_at vjepa2 204698b45b3712590f06245fbfba32d3be539812 https://github.com/facebookresearch/vjepa2.git
clone_at navi 3fcfd406aba3dbd43760ad100afb04a6320b7d1c https://github.com/google/navi.git
clone_at dinov2 7764ea0f912e53c92e82eb78a2a1631e92725fc8 https://github.com/facebookresearch/dinov2.git
if ! command -v uv >/dev/null; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi
cd "$CODE"
uv sync --frozen --extra gpu
uv run --frozen --extra gpu python ucloud/prepare_upstream.py
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv > /work/results/device.csv
if [[ ! -f /work/data/navi/.extraction-complete ]]; then
  mkdir -p /work/data/navi
  curl -fL --retry 2 --max-time 1200 -C - https://storage.googleapis.com/gresearch/navi-dataset/navi_v1.5.tar.gz -o /work/cache/navi_v1.5.tar.gz
  tar -tzf /work/cache/navi_v1.5.tar.gz | head -n 8 > /work/results/navi-archive-layout.txt || true
  tar -xzf /work/cache/navi_v1.5.tar.gz -C /work/data/navi --skip-old-files
  touch /work/data/navi/.extraction-complete
fi
uv run --frozen --extra gpu python -m video3d.cli prepare
GROUP="$(uv run --frozen --extra gpu python ucloud/resolve_group.py resolve)"
if [[ "$(uv run --frozen --extra gpu python ucloud/resolve_group.py profile-needed "$GROUP")" == True ]]; then
  timeout --signal=INT --kill-after=30 300 uv run --frozen --extra gpu python -m video3d.cli profile "training.group=$GROUP"
fi
uv run --frozen --extra gpu python -m video3d.cli sweep "training.group=$GROUP"
uv run --frozen --extra gpu python ucloud/resolve_group.py complete
PIPELINE
