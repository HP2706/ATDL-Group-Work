#!/usr/bin/env bash
# First run on a new node: does Blender render on this GPU, are the clips valid, and how many
# workers per GPU are fastest. Uses GPU 0 only and writes to a throwaway dataset root.
#
# Usage: bash scripts/generation/m2/smoke_test.sh [SPLIT] [SCRATCH]
#   SPLIT defaults to train (needs dataset/assets/train); eval works with the GSO assets.
set -euo pipefail

SPLIT=${1:-train}
SCRATCH=${2:-/tmp/video3d.data.generation_smoke}
CONFIG=${CONFIG:-configs/generation/m2.json}
root=$SCRATCH/dataset

if [[ -e "$SCRATCH" ]]; then
    mv "$SCRATCH" "${SCRATCH}.preserved-$(date +%s)-$$"
fi
mkdir -p "$root"
ln -s "$(realpath dataset/assets)" "$root/assets"
export CUDA_VISIBLE_DEVICES=0
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

render() {  # render <workers> <first seed> <seeds per worker>
    local workers=$1 first=$2 per=$3 pids=()
    local t0=$(date +%s)
    for ((w = 0; w < workers; w++)); do
        uv run python -m video3d.data.generation.generate --config "$CONFIG" --split "$SPLIT" --dataset-root "$root" \
            --seeds "[$(seq $((first + w * per)) $((first + (w + 1) * per - 1)) | paste -sd, -)]" \
            > "$SCRATCH/w${workers}_${w}.log" 2>&1 &
        pids+=($!)
    done
    for pid in "${pids[@]}"; do wait "$pid"; done
    local secs=$(( $(date +%s) - t0 ))
    echo "$workers worker(s): $((workers * per)) clips in ${secs}s = $(echo "scale=1; $secs / ($workers * $per)" | bc) s/clip"
}

# The first render also installs BlenderProc's packages into Blender and compiles kernels.
uv run python -m video3d.data.generation.generate --config "$CONFIG" --split "$SPLIT" --dataset-root "$root" \
    --seeds 1000000 > "$SCRATCH/warmup.log" 2>&1 || { tail -50 "$SCRATCH/warmup.log"; exit 1; }
render 1 1000100 4
render 2 1000200 4
render 4 1000300 4

python3 - "$root" "$SPLIT" <<'EOF'
import json, pathlib, sys
metas = [json.loads(p.read_text()) for p in pathlib.Path(sys.argv[1], "clips", sys.argv[2]).glob("*/meta.json")]
print(f"{len(metas)} clips, blender {sorted({m['render'].get('blender_version') for m in metas})}, "
      f"retries in {sum(bool(m['rejections']) for m in metas)} clips")
EOF
cat "$root/failures.jsonl" 2>/dev/null || true
SAM3DVID_DATASET=$root uv run pytest -q tests/generation -m "not slow" -p no:cacheprovider
echo "Smoke test passed. Pick WORKERS_PER_GPU from the fastest s/clip above."
