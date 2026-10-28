#!/usr/bin/env bash
# Build the image with Cloud Build and push it to Artifact Registry:  scripts/gcp/build.sh
# The upload respects .gitignore (no .venv, data, outputs or third_party).
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=/dev/null
source "$HERE/gcp.env"
IMAGE_URI="$REGION-docker.pkg.dev/$PROJECT_ID/$REPOSITORY/$IMAGE_NAME:$IMAGE_TAG"

if ! gcloud artifacts repositories describe "$REPOSITORY" --project "$PROJECT_ID" \
        --location "$REGION" >/dev/null 2>&1; then
    gcloud artifacts repositories create "$REPOSITORY" --project "$PROJECT_ID" \
        --location "$REGION" --repository-format docker
fi
gcloud builds submit "$HERE/../.." --project "$PROJECT_ID" \
    --config "$HERE/cloudbuild.yaml" --substitutions "_IMAGE_URI=$IMAGE_URI" \
    --machine-type "$BUILD_MACHINE" --disk-size 200 --timeout 4h
echo "pushed $IMAGE_URI"
