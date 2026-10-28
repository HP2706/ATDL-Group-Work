#!/usr/bin/env bash
# Fetch what the code needs but the repo does not ship: the pinned SAM 3D and V-JEPA 2 checkouts
# and the gated SAM 3D Stage-1 checkpoints (needs an HF token with access to facebook/sam-3d-objects).
#   bash scripts/fetch_upstream.sh --repos-only   # repos only
#   bash scripts/fetch_upstream.sh                # repos + checkpoints (prompts for the token)
# Existing checkouts are reused only at the pinned commit. The token is never stored.
set -euo pipefail

PROJECT_DIR="${PROJECT_DIR:-/work/vidsam3d}"
SAM3D_REPO="${SAM3D_REPO:-$PROJECT_DIR/sam-3d-objects}"
VJEPA2_REPO="${VJEPA2_REPO:-$PROJECT_DIR/vjepa2}"
SAM3D_CHECKPOINT_DIR="${SAM3D_CHECKPOINT_DIR:-$PROJECT_DIR/checkpoints/sam3d}"
SAM3D_COMMIT="f91db411c50efee93d8db7aeb323885650f6f722"
VJEPA2_COMMIT="204698b45b3712590f06245fbfba32d3be539812"
DOWNLOAD_DIR="$PROJECT_DIR/.sam3d-checkpoints-download"

if ! command -v git >/dev/null 2>&1; then
    echo "git is required; install it first." >&2
    exit 1
fi
mkdir -p "$PROJECT_DIR"

checkout_pinned_repo() {
    local url="$1"
    local path="$2"
    local commit="$3"

    if [[ -d "$path/.git" ]]; then
        local current
        current="$(git -C "$path" rev-parse HEAD)"
        if [[ "$current" != "$commit" ]]; then
            echo "$path is at $current, expected pinned commit $commit; refusing to alter it." >&2
            exit 1
        fi
        echo "Using existing pinned checkout: $path"
    elif [[ -e "$path" ]]; then
        echo "$path exists but is not a Git checkout; move it or set the corresponding *_REPO path." >&2
        exit 1
    else
        git clone "$url" "$path"
        git -C "$path" checkout "$commit"
    fi
}

checkout_pinned_repo \
    https://github.com/facebookresearch/sam-3d-objects.git "$SAM3D_REPO" "$SAM3D_COMMIT"
checkout_pinned_repo \
    https://github.com/facebookresearch/vjepa2.git "$VJEPA2_REPO" "$VJEPA2_COMMIT"

if [[ "${1:-}" == "--repos-only" ]]; then
    printf 'Pinned upstream repos are ready.\n'
    printf 'SAM3D_REPO=%s\nVJEPA2_REPO=%s\n' "$SAM3D_REPO" "$VJEPA2_REPO"
    exit 0
elif [[ $# -gt 0 ]]; then
    echo "Usage: $0 [--repos-only]" >&2
    exit 2
fi

if ! command -v hf >/dev/null 2>&1; then
    echo "The Hugging Face CLI is required; run: python -m pip install 'huggingface_hub[cli]'" >&2
    exit 1
fi

if [[ -z "${HF_TOKEN:-}" ]]; then
    read -rsp "Hugging Face read token: " HF_TOKEN
    printf '\n'
    export HF_TOKEN
fi
trap 'unset HF_TOKEN' EXIT

mkdir -p "$SAM3D_CHECKPOINT_DIR" "$DOWNLOAD_DIR"
hf download --repo-type model \
    --local-dir "$DOWNLOAD_DIR" \
    --max-workers 1 \
    facebook/sam-3d-objects \
    checkpoints/pipeline.yaml \
    checkpoints/ss_generator.yaml checkpoints/ss_generator.ckpt \
    checkpoints/ss_decoder.yaml checkpoints/ss_decoder.ckpt \
    checkpoints/ss_encoder.yaml checkpoints/ss_encoder.ckpt

for name in pipeline.yaml ss_generator.yaml ss_generator.ckpt \
    ss_decoder.yaml ss_decoder.ckpt ss_encoder.yaml ss_encoder.ckpt; do
    mv "$DOWNLOAD_DIR/checkpoints/$name" "$SAM3D_CHECKPOINT_DIR/$name"
done

for name in pipeline.yaml ss_generator.ckpt ss_decoder.ckpt ss_encoder.ckpt; do
    test -s "$SAM3D_CHECKPOINT_DIR/$name"
done
test -f "$VJEPA2_REPO/src/hub/backbones.py"

printf 'Pinned upstream repos and SAM 3D checkpoints are ready.\n'
printf 'SAM3D_REPO=%s\nVJEPA2_REPO=%s\nSAM3D_CHECKPOINT_DIR=%s\n' \
    "$SAM3D_REPO" "$VJEPA2_REPO" "$SAM3D_CHECKPOINT_DIR"
