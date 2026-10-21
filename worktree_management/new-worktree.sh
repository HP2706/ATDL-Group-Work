#!/usr/bin/env bash
set -Eeuo pipefail

BASE="main"
BRANCH=""
BOOTSTRAP=1

while [[ $# -gt 0 ]]; do
  case "$1" in
    --from)
      if [[ $# -lt 2 ]]; then
        echo "--from requires a base branch." >&2
        exit 1
      fi
      BASE="$2"
      shift 2
      ;;
    --no-bootstrap)
      BOOTSTRAP=0
      shift
      ;;
    -h | --help)
      echo "Usage: $0 <branch> [--from <base>] [--no-bootstrap]"
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
  echo "Usage: $0 <branch> [--from <base>] [--no-bootstrap]" >&2
  exit 1
fi

git check-ref-format --branch "$BRANCH" >/dev/null
LOWER_BRANCH="$(printf '%s' "$BRANCH" | tr '[:upper:]' '[:lower:]')"
case "$LOWER_BRANCH" in
  *codex* | *claude* | *cursor*)
    echo "Refusing an agent-attributed branch name: $BRANCH" >&2
    exit 1
    ;;
esac

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
GIT_COMMON_DIR="$(git -C "$SCRIPT_DIR" rev-parse --path-format=absolute --git-common-dir)"
PRIMARY_ROOT="$(git --git-dir="$GIT_COMMON_DIR" rev-parse --show-toplevel)"
WORKTREE_DIR="$PRIMARY_ROOT/.worktrees/$BRANCH"
ENVIRONMENT_LIST="$PRIMARY_ROOT/worktree_management/environment-files.txt"
CREATE_HOOK="$PRIMARY_ROOT/worktree_management/after-create.sh"

if [[ ! -f "$ENVIRONMENT_LIST" ]]; then
  echo "Missing $ENVIRONMENT_LIST" >&2
  exit 1
fi

while IFS= read -r line || [[ -n "$line" ]]; do
  rel="${line#"${line%%[![:space:]]*}"}"
  rel="${rel%"${rel##*[![:space:]]}"}"
  [[ -z "$rel" || "$rel" == \#* ]] && continue
  optional=0
  if [[ "$rel" == \?* ]]; then
    optional=1
    rel="${rel#\?}"
  fi
  case "$rel" in
    /* | .. | ../* | */../* | */.. | .git | .git/*)
      echo "Unsafe environment file path: $rel" >&2
      exit 1
      ;;
  esac
  if [[ ! -f "$PRIMARY_ROOT/$rel" && "$optional" -eq 0 ]]; then
    echo "Required environment file is missing from the primary checkout: $rel" >&2
    exit 1
  fi
done < "$ENVIRONMENT_LIST"

if [[ -e "$WORKTREE_DIR" ]]; then
  if ! git -C "$PRIMARY_ROOT" worktree list --porcelain |
    sed -n 's/^worktree //p' |
    grep -Fqx "$WORKTREE_DIR"; then
    echo "Target exists but is not a registered worktree: $WORKTREE_DIR" >&2
    exit 1
  fi
  CURRENT_BRANCH="$(git -C "$WORKTREE_DIR" branch --show-current)"
  if [[ "$CURRENT_BRANCH" != "$BRANCH" ]]; then
    echo "Target worktree uses '$CURRENT_BRANCH', not '$BRANCH'." >&2
    exit 1
  fi
  echo "Reusing registered worktree: $WORKTREE_DIR"
else
  mkdir -p "$(dirname "$WORKTREE_DIR")"
  if git -C "$PRIMARY_ROOT" show-ref --verify --quiet "refs/heads/$BRANCH"; then
    git -C "$PRIMARY_ROOT" worktree add "$WORKTREE_DIR" "$BRANCH"
  elif git -C "$PRIMARY_ROOT" show-ref --verify --quiet "refs/remotes/origin/$BRANCH"; then
    git -C "$PRIMARY_ROOT" worktree add \
      "$WORKTREE_DIR" -b "$BRANCH" "origin/$BRANCH"
  else
    if git -C "$PRIMARY_ROOT" show-ref --verify --quiet "refs/heads/$BASE"; then
      BASE_REF="$BASE"
    elif git -C "$PRIMARY_ROOT" show-ref --verify --quiet "refs/remotes/origin/$BASE"; then
      BASE_REF="origin/$BASE"
    else
      echo "Base branch '$BASE' does not exist locally or on origin." >&2
      exit 1
    fi
    git -C "$PRIMARY_ROOT" worktree add \
      "$WORKTREE_DIR" -b "$BRANCH" "$BASE_REF"
  fi

  if [[ -f "$WORKTREE_DIR/.gitmodules" ]]; then
    git -C "$WORKTREE_DIR" submodule update --init --recursive
  fi
fi

echo "Synchronizing ignored environment files..."
while IFS= read -r line || [[ -n "$line" ]]; do
  rel="${line#"${line%%[![:space:]]*}"}"
  rel="${rel%"${rel##*[![:space:]]}"}"
  [[ -z "$rel" || "$rel" == \#* ]] && continue
  rel="${rel#\?}"
  [[ -f "$PRIMARY_ROOT/$rel" ]] || continue
  mkdir -p "$(dirname "$WORKTREE_DIR/$rel")"
  cp -p "$PRIMARY_ROOT/$rel" "$WORKTREE_DIR/$rel"
  chmod go-rwx "$WORKTREE_DIR/$rel"
  echo "  $rel"
done < "$ENVIRONMENT_LIST"

if [[ "$BOOTSTRAP" -eq 1 ]]; then
  if [[ ! -f "$CREATE_HOOK" ]]; then
    echo "Missing creation hook: $CREATE_HOOK" >&2
    exit 1
  fi
  bash "$CREATE_HOOK" "$PRIMARY_ROOT" "$WORKTREE_DIR" "$BRANCH"
fi

echo "Worktree ready: $WORKTREE_DIR"
