#!/usr/bin/env bash
set -euo pipefail

# UCloud "Batch processing" script (see UCLOUD_ABLATIONS.md). Upload it next to
# ablations-ucloud-code.zip and attach that folder to the job. It unpacks the zip
# when needed, then runs every ablation; the job ends when this script finishes.

shopt -s nullglob
search_root="${1:-/work}"
zips=("$search_root"/*/ablations-ucloud-code.zip "$search_root"/*/*/ablations-ucloud-code.zip)
if [[ ${#zips[@]} -ne 1 ]]; then
  echo "Expected exactly one ablations-ucloud-code.zip in the attached folder; found ${#zips[@]}." >&2
  exit 1
fi
drive="$(dirname "${zips[0]}")"
helper="$drive/ablation-code/ATDL-Group-Work/Assignment 2 - Group/code/scripts/ablations_direct.sh"
if [[ ! -f "$helper" || "${zips[0]}" -nt "$helper" ]]; then
  if command -v unzip >/dev/null; then
    unzip -oq "${zips[0]}" -d "$drive/ablation-code"
  else
    python3 -m zipfile -e "${zips[0]}" "$drive/ablation-code"
  fi
  touch "$helper"
fi
ATDL_WORK_ROOT="$drive" bash "$helper" 2>&1 | tee -a "$drive/ablations.log"
