#!/usr/bin/env bash
set -uo pipefail

# Run the targeted ablations inside a UCloud GPU job (see UCLOUD_ABLATIONS.md).
# Usage: bash ablations_direct.sh [setup] [x1] [x2] [x3] [x4] [package]   (default: all of them)
# Each ablation has a fixed run root, so running the same target again resumes it.
# Without the team's baseline sweeps on the drive, X1 trains from scratch and X2 adds seed 0.

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_dir="$(cd "$script_dir/../.." && pwd)"
repo_dir="$(cd "$project_dir/.." && pwd)"
work_root="${ATDL_WORK_ROOT:-$(printf '%s' "$script_dir" | cut -d/ -f1-3)}"
if [[ "$work_root" != /work/?* || ! -d "$work_root" ]]; then
  echo 'Set ATDL_WORK_ROOT to the mounted drive, for example /work/my-drive.' >&2
  exit 1
fi
export ATDL_WORK_ROOT="$work_root"
export PYTHON_BIN="${PYTHON_BIN:-$work_root/.venv/bin/python}"
export ATDL_LAUNCH_CONFIG="$(mktemp)"
printf 'backend = "direct"\n' > "$ATDL_LAUNCH_CONFIG"
if [[ -z "${WANDB_API_KEY:-}" ]]; then
  export WANDB_MODE=disabled
fi
sweeps="$work_root/runs/sweeps/ablations"
baseline_resnet="$work_root/runs/sweeps/resnet-cifar10-adam-400-2026-09-26-21-52-00"
baseline_cnn="$work_root/runs/sweeps/cnn-cifar10-sgd-50000-2026-09-27-10-37-31"
failed=""

setup() {
  if [[ ! -x "$PYTHON_BIN" ]]; then
    echo "=== $(date '+%F %T') creating the Python environment in $work_root/.venv"
    if ! command -v uv >/dev/null; then
      curl -LsSf https://astral.sh/uv/install.sh | env UV_INSTALL_DIR="$work_root/.uv/bin" INSTALLER_NO_MODIFY_PATH=1 sh || return 1
      export PATH="$work_root/.uv/bin:$PATH"
    fi
    UV_PROJECT_ENVIRONMENT="$work_root/.venv" UV_PYTHON_INSTALL_DIR="$work_root/.uv/python" UV_LINK_MODE=copy UV_PYTHON_PREFERENCE=only-managed \
      uv sync --frozen --no-install-project --project "$repo_dir" || return 1
  fi
  "$PYTHON_BIN" -c "import chz, fire, pandas, pyarrow, pydantic, torch, torchvision, tqdm, wandb
assert torch.cuda.is_available(), 'no CUDA GPU visible'
print('torch', torch.__version__, 'on', torch.cuda.get_device_name(0))" || return 1
  (cd "$project_dir/code" && "$PYTHON_BIN" -c "from pathlib import Path; from sweep import prepare_cifar; prepare_cifar('cifar10', Path('$work_root/data'))") || return 1
  echo "=== $(date '+%F %T') setup done"
}

run() {
  local label=$1 script=$2
  shift 2
  echo "=== $(date '+%F %T') start $label (run root $sweeps/$label)"
  if (cd "$project_dir" && bash "code/scripts/$script.sh" "run_root=$sweeps/$label" "$@"); then
    echo "=== $(date '+%F %T') done $label"
  else
    echo "=== $(date '+%F %T') FAILED or interrupted $label; run the same target again to resume"
    failed="$failed $label"
  fi
}

# Continue the team's baseline runs when they are on this drive; otherwise train from scratch.
x1() {
  if [[ -d "$baseline_resnet" ]]; then run x1-horizon-resnet x1_horizon_resnet; else run x1-horizon-resnet x1_horizon_resnet extend_from=; fi
  if [[ -d "$baseline_cnn" ]]; then run x1-horizon-cnn x1_horizon_cnn; else run x1-horizon-cnn x1_horizon_cnn extend_from=; fi
}

x2() {
  if [[ ! -d "$baseline_resnet" ]]; then
    run x2-seeds-seed0 x2_seeds seed=0
  fi
  run x2-seeds-seed1 x2_seeds seed=1
  run x2-seeds-seed2 x2_seeds seed=2
}

package() {
  "$PYTHON_BIN" - "$work_root" "$sweeps" "$baseline_resnet" <<'EOF'
import sys
import zipfile
from pathlib import Path

root, sweeps, baseline = map(Path, sys.argv[1:])
results = root / "our-results-folder"
files = {path: path.relative_to(root) for path in sorted((results / "ablations").rglob("*.parquet"))}
predictions = {"seed0": sweeps / "x2-seeds-seed0" if (sweeps / "x2-seeds-seed0").exists() else baseline,
               "seed1": sweeps / "x2-seeds-seed1", "seed2": sweeps / "x2-seeds-seed2"}
for seed, sweep in predictions.items():
    for path in sorted(sweep.glob("noise-20/width-*/*/test_predictions.pt")):
        files[path] = Path("our-results-folder/ablations/x2-seeds/predictions") / seed / path.parts[-3] / path.name
archive = root / "ablation-results.zip"
with zipfile.ZipFile(archive, "w") as output:
    for path, name in files.items():
        output.write(path, name)
print(f"Packed {len(files)} files into {archive}")
EOF
}

if [[ $# -eq 0 ]]; then
  set -- setup x1 x2 x3 x4 package
fi
for target in "$@"; do
  case "$target" in
    setup) setup || { echo "Setup failed; see the messages above." >&2; exit 1; } ;;
    x1) x1 ;;
    x2) x2 ;;
    x3) run x3-weight-decay x3_weight_decay ;;
    x4) run x4-optimizer x4_optimizer ;;
    package) package ;;
    *) echo "Unknown target: $target (use setup, x1, x2, x3, x4, or package)" >&2; exit 1 ;;
  esac
done
rm -f "$ATDL_LAUNCH_CONFIG"
if [[ -n "$failed" ]]; then
  echo "Not finished:$failed"
  exit 1
fi
echo "All requested targets finished."
