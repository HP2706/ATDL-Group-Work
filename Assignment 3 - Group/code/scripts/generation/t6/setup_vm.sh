#!/usr/bin/env bash
set -Eeuo pipefail
CODE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
exec bash "$CODE_ROOT/scripts/setup_env.sh"
