#!/usr/bin/env bash
# Run one command on any platform:  scripts/run.sh <command> [args...]
#   -> python -m vidsam3d <command> [args...] with configs/env/$VIDSAM3D_ENV.yaml
# 1. activates the env from setup_env.sh (ACTIVATE=0: use the python already on PATH)
# 2. puts SCRATCH_ROOT on job-local disk if it has >= MIN_LOCAL_SCRATCH_GB free (default 100)
# 3. if data.stage_to_scratch (or STAGE_TO_SCRATCH=1): copies the export folders the command
#    reads to scratch and points DATA_ROOT there (random access to shards on network storage is slow)
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

# export folders the command reads: --split if given, else the command's default (val is in train)
split_arg=""
prev=""
for arg in "$@"; do
    [[ "$prev" == "--split" ]] && split_arg="$arg"
    prev="$arg"
done
case "$cmd" in
    train) folders="$(cfg data.train_split)" ;;
    eval | frame-test | inspect-sam3d) folders="${split_arg:-$(cfg data.eval_split)}" ;;
    cache-latents | validate-data) folders="$split_arg" ;;
    *) folders="" ;;
esac
[[ "$folders" == val ]] && folders="$(cfg data.train_split)"

copy_dir() {   # copy_dir <src> <dst>; gs:// directly when the source is a GCS FUSE path
    mkdir -p "$2"
    if [[ "$1" == /gcs/* ]] && command -v gcloud >/dev/null; then
        gcloud storage rsync --recursive "gs://${1#/gcs/}" "$2"
    else
        rsync -a "$1/" "$2/"
    fi
}

stage="${STAGE_TO_SCRATCH:-$(cfg data.stage_to_scratch)}"
if [[ "$stage" =~ ^(1|true|True)$ && -n "$folders" ]]; then
    src="$(cfg paths.data_root)"
    dst="$(cfg paths.scratch_root)/dataset"
    for folder in $folders; do
        echo "staging $src/$folder -> $dst/$folder"
        copy_dir "$src/$folder" "$dst/$folder"
    done
    export DATA_ROOT="$dst"
fi

exec python -m vidsam3d "$cmd" "$@"
