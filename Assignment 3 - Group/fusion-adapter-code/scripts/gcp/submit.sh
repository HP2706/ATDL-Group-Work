#!/usr/bin/env bash
# Submit a Vertex AI custom job:  scripts/gcp/submit.sh <command> [args...]
# e.g. scripts/gcp/submit.sh train --condition c3
# The container runs scripts/run.sh <command> [args...] with VIDSAM3D_ENV=gcp. HF_TOKEN comes from
# $HF_TOKEN or Secret Manager ($HF_SECRET) and is only needed if checkpoints must be downloaded.
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 <command> [args...]" >&2; exit 2; }
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/gcp.env"
for var in PROJECT_ID REGION GCS_BUCKET SERVICE_ACCOUNT MACHINE_TYPE ACCELERATOR_TYPE; do
    [[ -n "${!var:-}" ]] || { echo "set $var in $HERE/gcp.env" >&2; exit 1; }
done
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG"
if [[ -z "${HF_TOKEN:-}" && -n "${HF_SECRET:-}" ]]; then
    HF_TOKEN="$(gcloud secrets versions access latest --secret "$HF_SECRET" --project "$PROJECT_ID")"
fi

config="$(mktemp)"
trap 'rm -f "$config"' EXIT
chmod 600 "$config"
# JSON is valid YAML; python does the quoting of the command arguments
HF_TOKEN="${HF_TOKEN:-}" python3 - "$@" > "$config" <<PY
import json, os, sys
env = {"VIDSAM3D_ENV": "gcp", "GCS_BUCKET": "$GCS_BUCKET",
       "HF_HOME": "/gcs/$GCS_BUCKET/hf_home", "TORCH_HOME": "/gcs/$GCS_BUCKET/torch_home"}
if os.environ["HF_TOKEN"]:
    env["HF_TOKEN"] = os.environ["HF_TOKEN"]
spec = {"workerPoolSpecs": [{
    "machineSpec": {"machineType": "$MACHINE_TYPE", "acceleratorType": "$ACCELERATOR_TYPE",
                    "acceleratorCount": int("$ACCELERATOR_COUNT")},
    "replicaCount": 1,
    "containerSpec": {"imageUri": "$IMAGE_URI", "args": sys.argv[1:],
                      "env": [{"name": k, "value": v} for k, v in env.items()]}}],
    "serviceAccount": "$SERVICE_ACCOUNT"}
print(json.dumps(spec))
PY
gcloud ai custom-jobs create --project "$PROJECT_ID" --region "$REGION" \
    --display-name "vidsam3d-$1-$(date +%Y%m%d-%H%M%S)" --config "$config"
