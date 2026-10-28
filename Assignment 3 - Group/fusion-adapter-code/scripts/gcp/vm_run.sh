#!/usr/bin/env bash
# Option B, a plain GPU VM:  scripts/gcp/vm_run.sh <command> [args...]
# Mounts the bucket at /gcs/$GCS_BUCKET with gcsfuse, then runs like UCloud. Build the env once on
# the VM with scripts/fetch_upstream.sh and scripts/setup_env.sh (PROJECT_DIR default ~/vidsam3d).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/gcp.env"
mount="/gcs/$GCS_BUCKET"
if ! mountpoint -q "$mount"; then
    sudo mkdir -p "$mount"
    sudo chown "$(id -u):$(id -g)" "$mount"
    gcsfuse --implicit-dirs "$GCS_BUCKET" "$mount"
fi
export PROJECT_DIR="${PROJECT_DIR:-$HOME/vidsam3d}" VIDSAM3D_ENV=gcp GCS_BUCKET
exec "$HERE/../run.sh" "$@"
