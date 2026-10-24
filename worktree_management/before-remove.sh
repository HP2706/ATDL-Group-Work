#!/usr/bin/env bash
set -Eeuo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 <primary-root> <worktree-path> <branch>" >&2
  exit 1
fi

PRIMARY_ROOT="$1"
WORKTREE_DIR="$2"
BRANCH="$3"

for directory in "$WORKTREE_DIR/Assignment 3 - Group/code/data" "$WORKTREE_DIR/Assignment 3 - Group/code/results" "$WORKTREE_DIR/Assignment 3 - Group/code/dependencies"; do
  if [[ -d "$directory" && -n "$(ls -A "$directory")" ]]; then
    echo "Preserve cached/generated experiment artifacts before cleanup: $directory" >&2
    exit 1
  fi
done

# Add the exact inverse of after-create.sh here. Stop and remove only resources
# whose validated names are deterministically owned by this worktree. This hook
# runs before the worktree and its ignored environment files are removed.
#
# Example shape:
# SAFE_ID="$(printf '%s' "$BRANCH" | tr -cs '[:alnum:]' '_' | tr '[:upper:]' '[:lower:]')"
# DATABASE_NAME="project_worktree_${SAFE_ID}"
# [[ "$DATABASE_NAME" == project_worktree_* ]]
# dropdb --if-exists "$DATABASE_NAME"

printf 'No project-specific resources configured for %s at %s (primary: %s).\n' \
  "$BRANCH" "$WORKTREE_DIR" "$PRIMARY_ROOT"
