#!/usr/bin/env bash
# Submit this as Terminal Ubuntu's batchScript. Its foreground exit releases CPU.
set -euo pipefail
task_root="${1:-/work/MathiasElmegaardNielsen#6181/omni-gso-data}"
code_root="$task_root/repo/Assignment 3 - Group/code"
mkdir -p "$task_root/logs" "$task_root/runtime"
exec > >(tee -a "$task_root/logs/pipeline.log") 2>&1
finish() {
  result=$?
  printf '{"exit_code":%d,"finished_utc":"%s"}\n' "$result" "$(date -u +%FT%TZ)" > "$task_root/logs/pipeline-exit.json"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
export UV_CACHE_DIR="$task_root/runtime/uv-cache"
export UV_PYTHON_INSTALL_DIR="$task_root/runtime/python"
export UV_PROJECT_ENVIRONMENT="$task_root/runtime/cpu-venv"
export UV_LINK_MODE=copy
sudo apt-get update -qq
sudo apt-get install -y -qq aria2 libgl1 libegl1 libgl1-mesa-dri libglu1-mesa libxrender1 ffmpeg build-essential > "$task_root/logs/system.log" 2>&1
uv_binary="$task_root/runtime/uv"
cd "$code_root"
git rev-parse HEAD > "$task_root/logs/source-commit.txt"
"$uv_binary" sync --inexact --group dev --group generation > "$task_root/logs/environment.log" 2>&1
python_binary="$UV_PROJECT_ENVIRONMENT/bin/python"
export PATH="$task_root/runtime:$PATH"
"$python_binary" - <<'PY'
import sys,tomllib
sys.path.insert(0, 'scripts')
from install_models import install_archive
with open('pyproject.toml', 'rb') as stream:
    revision = tomllib.load(stream)['tool']['video3d']['upstream']['vjepa2']
install_archive('facebookresearch/vjepa2', revision, 'atdl-vjepa2', packages=['src', 'app'])
PY
"$python_binary" -m pytest tests/test_scanned.py tests/test_pooler.py tests/test_config.py tests/test_adapter.py -q > "$task_root/logs/tests.log" 2>&1
"$python_binary" scripts/download_gso.py --out "$task_root/data/raw/gso" --workers 4 > "$task_root/logs/gso-download.log" 2>&1
"$python_binary" -m video3d scanned prepare-gso --root "$task_root/data/raw/gso" --output "$task_root/data/gso" --workers 3 --frames 32 --resolution 384 > "$task_root/logs/gso-prepare.log" 2>&1
"$python_binary" scripts/check_scanned.py --root "$task_root/data/gso" --source gso > "$task_root/logs/gso-check.log" 2>&1
"$python_binary" scripts/download_omniobject.py --out "$task_root/data/raw/omniobject3d" --workers 6 > "$task_root/logs/omni-download.log" 2>&1
"$python_binary" -m video3d scanned prepare-omni --root "$task_root/data/raw/omniobject3d" --output "$task_root/data/omniobject3d" --workers 3 > "$task_root/logs/omni-prepare.log" 2>&1
"$python_binary" scripts/check_scanned.py --root "$task_root/data/omniobject3d" --source omniobject3d > "$task_root/logs/omni-check.log" 2>&1
TASK_DATA_ROOT="$task_root/data" "$python_binary" - <<'PY'
import json,os
from pathlib import Path
root = Path(os.environ['TASK_DATA_ROOT'])
omni = json.loads((root/'omniobject3d/preparation.json').read_text())
gso = json.loads((root/'raw/gso/index.json').read_text())
if omni['missing_scan_objects']:
    raise ValueError('Some captured objects have no corresponding scan; see preparation.json')
if gso['num_downloaded'] != gso['num_models']:
    raise ValueError('Incomplete GSO download')
result = {'omniobject3d': omni, 'gso_objects':gso['num_downloaded'],
          'loader_checks':'passed', 'pretrained_gpu_validation':'pending',
          'omni_scan_camera_registration':'not_verified'}
(root/'completion.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
PY
