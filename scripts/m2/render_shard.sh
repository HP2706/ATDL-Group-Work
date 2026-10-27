#!/usr/bin/env bash
# Render train seeds [START, END) on all GPUs of this node. Run from the repo root.
#
# Usage: bash scripts/m2/render_shard.sh START END [DATASET_ROOT] [WORKERS_PER_GPU]
#   e.g. bash scripts/m2/render_shard.sh 1000000 1002500 /scratch/m2/dataset 2
#
# Seeds are dealt round-robin to GPUS x WORKERS_PER_GPU processes. Finished clips are skipped,
# so a killed job is resumed by running the same command again. Logs: DATASET_ROOT/logs/.
set -euo pipefail

START=$1
END=$2
DATASET_ROOT=${3:-dataset}
WORKERS_PER_GPU=${4:-2}
CONFIG=${CONFIG:-configs/m2.yaml}

NUM_GPUS=$(nvidia-smi -L | wc -l)
WORKERS=$((NUM_GPUS * WORKERS_PER_GPU))
mkdir -p "$DATASET_ROOT/logs"
echo "seeds [$START, $END) on $NUM_GPUS GPUs x $WORKERS_PER_GPU workers -> $DATASET_ROOT"

pids=()
for ((w = 0; w < WORKERS; w++)); do
    gpu=$((w % NUM_GPUS))
    seeds=$(seq $((START + w)) "$WORKERS" $((END - 1)))
    [ -z "$seeds" ] && continue
    CUDA_VISIBLE_DEVICES=$gpu uv run python -m sam3dvid.generate --config "$CONFIG" --split train \
        --dataset-root "$DATASET_ROOT" --seeds $seeds \
        > "$DATASET_ROOT/logs/shard_${START}_${END}_w${w}.log" 2>&1 &
    pids+=($!)
done

status=0
for pid in "${pids[@]}"; do wait "$pid" || status=1; done
done_count=$(find "$DATASET_ROOT/clips/train" -maxdepth 2 -name meta.json 2>/dev/null | wc -l)
# failures.jsonl only exists once a seed has failed; under pipefail a missing file would abort here.
failed=0
[ -f "$DATASET_ROOT/failures.jsonl" ] && failed=$(wc -l < "$DATASET_ROOT/failures.jsonl")
echo "finished: $done_count train clips in $DATASET_ROOT, $failed failed seeds logged in total"
exit $status
