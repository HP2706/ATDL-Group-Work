#!/usr/bin/env bash
set -Eeuo pipefail
umask 077
# Install this as the application's foreground Batch processing entrypoint.
# Process exit must end the provider allocation; do not use SSH/nohup reservations.
RUN_ROOT="${RUN_ROOT:-/work/hp-video}"
mkdir -p "$RUN_ROOT"
exec 9>"$RUN_ROOT/job.lock"
flock -n 9 || { echo 'Another batch job owns this staging directory.' >&2; exit 1; }
exec > >(tee -a "$RUN_ROOT/batch.log") 2>&1
finish() {
  status=$?
  printf '{"exit_code":%s,"finished_at":"%s"}\n' "$status" "$(date -u +%FT%TZ)" > "$RUN_ROOT/job-exit.json"
  sync
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
set -a
. "$RUN_ROOT/runtime.env"
set +a
: "${HP_SOURCE_COMMIT:?Pin the reviewed source commit}"
: "${RUN_CONFIG:?Set the absolute path to a resolved experiment JSON}"
export HF_HOME="$RUN_ROOT/cache/huggingface" TORCH_HOME="$RUN_ROOT/cache/torch"
export HF_TOKEN="${HF_TOKEN:-${HF_READ_TOKEN:-}}"
export WANDB_DIR="$RUN_ROOT/results/wandb" LIDRA_SKIP_INIT=1 ATTN_BACKEND=sdpa
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
mkdir -p "$WANDB_DIR" "$HF_HOME" "$TORCH_HOME"
checkout="$RUN_ROOT/source-$HP_SOURCE_COMMIT"
if [[ ! -d "$checkout/.git" ]]; then
  git clone --filter=blob:none --no-checkout https://github.com/HP2706/ATDL-Group-Work.git "$checkout"
  git -C "$checkout" fetch --depth 1 origin "$HP_SOURCE_COMMIT"
  git -C "$checkout" checkout --detach "$HP_SOURCE_COMMIT"
fi
if [[ "$(git -C "$checkout" rev-parse HEAD)" != "$HP_SOURCE_COMMIT" ]]; then
  echo 'Staged checkout revision mismatch.' >&2
  exit 1
fi
cd "$checkout/Assignment 3 - Group/code"
# A provider wall-time limit is still required. This foreground deadline leaves cleanup time.
export RUN_CONFIG
export BATCH_SWEEP="${BATCH_SWEEP:-0}"
timeout --signal=INT --kill-after=60 "${PIPELINE_SECONDS:-6900}" bash -s <<'PIPELINE'
set -Eeuo pipefail
bash scripts/setup_env.sh
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
if [[ "$BATCH_SWEEP" == 1 ]]; then
  python -c 'import os; from video3d.config import read_config, apply_process_env; from video3d.sweep import sweep; c=read_config(os.environ["RUN_CONFIG"]); apply_process_env(c); sweep(c)'
else
  python -m video3d train-json "$RUN_CONFIG"
  python -c 'import os; from pathlib import Path; from video3d.config import read_config, apply_process_env; from video3d.evaluate import evaluate; c=read_config(os.environ["RUN_CONFIG"]); apply_process_env(c); evaluate(c, str(Path(c.paths.output_root)/c.run_name))'
fi
PIPELINE
