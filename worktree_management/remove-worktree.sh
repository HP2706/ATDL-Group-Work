#!/usr/bin/env bash
set -Eeuo pipefail

BRANCH=""
KEEP_BRANCH=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --keep-branch)
      KEEP_BRANCH=1
      shift
      ;;
    -h | --help)
      echo "Usage: $0 <branch> [--keep-branch]"
      exit 0
      ;;
    *)
      if [[ -n "$BRANCH" ]]; then
        echo "Unexpected argument: $1" >&2
        exit 1
      fi
      BRANCH="$1"
      shift
      ;;
  esac
done

if [[ -z "$BRANCH" ]]; then
  echo "Usage: $0 <branch> [--keep-branch]" >&2
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_COMMON_DIR="$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir)"
PRIMARY_ROOT="$(git -C "$SCRIPT_DIR" rev-parse --show-toplevel)"
WORKTREE_DIR="$PRIMARY_ROOT/.worktrees/$BRANCH"
REMOVE_HOOK="$PRIMARY_ROOT/worktree_management/before-remove.sh"

if ! git -C "$PRIMARY_ROOT" worktree list --porcelain |
  sed -n 's/^worktree //p' |
  grep -Fqx "$WORKTREE_DIR"; then
  echo "Not a registered worktree: $WORKTREE_DIR" >&2
  exit 1
fi

CURRENT_BRANCH="$(git -C "$WORKTREE_DIR" branch --show-current)"
if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
  echo "Target worktree uses '$CURRENT_BRANCH', not '$BRANCH'." >&2
  exit 1
fi

if [[ -n "$(git -C "$WORKTREE_DIR" status --porcelain)" ]]; then
  echo "Refusing to remove a worktree with local changes: $WORKTREE_DIR" >&2
  git -C "$WORKTREE_DIR" status --short --branch >&2
  exit 1
fi

if [[ ! -f "$REMOVE_HOOK" ]]; then
  echo "Missing teardown hook: $REMOVE_HOOK" >&2
  exit 1
fi

bash "$REMOVE_HOOK" "$PRIMARY_ROOT" "$WORKTREE_DIR" "$BRANCH"
git -C "$PRIMARY_ROOT" worktree remove "$WORKTREE_DIR"

if [[ "$KEEP_BRANCH" -eq 1 ]]; then
  echo "Kept local branch $BRANCH."
  exit 0
fi

SAFE_TO_DELETE=0
if git -C "$PRIMARY_ROOT" merge-base --is-ancestor "$BRANCH" HEAD; then
  SAFE_TO_DELETE=1
elif git -C "$PRIMARY_ROOT" show-ref --verify --quiet \
  "refs/remotes/origin/$BRANCH" &&
  git -C "$PRIMARY_ROOT" merge-base --is-ancestor \
    "$BRANCH" "origin/$BRANCH"; then
  SAFE_TO_DELETE=1
fi

if [[ "$SAFE_TO_DELETE" -eq 1 ]]; then
  git -C "$PRIMARY_ROOT" branch -D "$BRANCH"
  echo "Removed worktree and deleted safely represented local branch $BRANCH."
else
  echo "Removed worktree but kept unmerged local branch $BRANCH."
fi
