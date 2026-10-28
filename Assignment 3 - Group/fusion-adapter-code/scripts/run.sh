#!/usr/bin/env bash
# Run one command on any platform:  scripts/run.sh <command> [args...]
#   -> python -m vidsam3d <command> [args...] with configs/env/$VIDSAM3D_ENV.yaml
# 1. activates the env from setup_env.sh (ACTIVATE=0: use the python already on PATH)
# 2. puts SCRATCH_ROOT on job-local disk if it has >= MIN_LOCAL_SCRATCH_GB free (default 100)
# 3. if data.stage_to_scratch (or STAGE_TO_SCRATCH=1): copies the splits the command reads to
#    scratch and points DATA_ROOT there (reading many small PNGs through a FUSE mount is slow)
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <command> [args...]" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [[ "${ACTIVATE:-1}" == 1 ]]; then
    set +u
    # shellcheck source=scripts/activate.sh
    source "$HERE/activate.sh"
    set -u
fi
export VIDSAM3D_ENV="${VIDSAM3D_ENV:-ucloud}"
cmd="$1"
shift

cfg() { python -c "from vidsam3d.config import load_config; print(load_config().$1)"; }

if [[ -z "${SCRATCH_ROOT:-}" ]]; then
    local_dir="${LOCAL_SCRATCH:-/tmp/vidsam3d-scratch}"
    mkdir -p "$local_dir"
    free_gb="$(df -Pk "$local_dir" | awk 'NR == 2 {print int($4 / 1048576)}')"
    if (( free_gb >= ${MIN_LOCAL_SCRATCH_GB:-100} )); then
        export SCRATCH_ROOT="$local_dir"
    fi                                       # else: the env file's default (persistent disk)
fi

# splits the command reads: --split if given, else the command's default
split_arg=""
prev=""
for arg in "$@"; do
    [[ "$prev" == "--split" ]] && split_arg="$arg"
    prev="$arg"
done
case "$cmd" in
    train) splits="$(cfg data.train_split) $(cfg data.val_split)" ;;
    eval | frame-test | inspect-sam3d) splits="${split_arg:-$(cfg data.eval_split)}" ;;
    cache-latents | validate-data) splits="$split_arg" ;;
    *) splits="" ;;
esac

copy_dir() {   # copy_dir <src> <dst>; gs:// directly when the source is a GCS FUSE path
    mkdir -p "$2"
    if [[ "$1" == /gcs/* ]] && command -v gcloud >/dev/null; then
        gcloud storage rsync --recursive "gs://${1#/gcs/}" "$2"
    else
        rsync -a "$1/" "$2/"
    fi
}

stage="${STAGE_TO_SCRATCH:-$(cfg data.stage_to_scratch)}"
if [[ "$stage" =~ ^(1|true|True)$ && -n "$splits" ]]; then
    src="$(cfg paths.data_root)"
    dst="$(cfg paths.scratch_root)/dataset"
    mkdir -p "$dst"
    cp "$src/manifest.jsonl" "$dst/manifest.jsonl"
    for split in $splits; do
        for sub in clips assets; do
            if [[ -d "$src/$sub/$split" ]]; then
                echo "staging $src/$sub/$split -> $dst/$sub/$split"
                copy_dir "$src/$sub/$split" "$dst/$sub/$split"
            fi
        done
    done
    export DATA_ROOT="$dst"
fi

exec python -m vidsam3d "$cmd" "$@"
