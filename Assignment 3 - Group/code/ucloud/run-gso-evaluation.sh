#!/usr/bin/env bash
set -euo pipefail
task_root="${1:?evaluation root required}"
method="${2:?model method required}"
code_root="$task_root/repo/Assignment 3 - Group/code"
mkdir -p "$task_root/logs" "$task_root/results/$method"
exec > >(tee -a "$task_root/logs/$method.log") 2>&1
finish() {
  status=$?
  printf '{"exit_code":%d,"finished_utc":"%s"}\n' "$status" "$(date -u +%FT%TZ)" > "$task_root/logs/$method-exit.json"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
test -f "$task_root/preparation-complete.json"
source /work/MathiasElmegaardNielsen\#6181/sam3dvid-pilot/common.sh
unset PYTHONPATH
export OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4
export UV_CACHE_DIR='/work/MathiasElmegaardNielsen#6181/sam3dvid-pilot/cache/uv'
export UV_LINK_MODE=copy
export UV_PROJECT_ENVIRONMENT="$task_root/runtime/$method"
export UV_PYTHON_INSTALL_DIR='/work/MathiasElmegaardNielsen#6181/omni-gso-data/runtime/python'
uv_binary='/work/MathiasElmegaardNielsen#6181/omni-gso-data/runtime/uv'
export PATH="$(dirname "$uv_binary"):$PATH"
extra=mvsam3d
if [ "$method" = stream3d ]; then extra=stream3d; fi
cd "$code_root"
"$uv_binary" sync --locked --extra "$extra" --group gpu_runtime
python_binary="$UV_PROJECT_ENVIRONMENT/bin/python"
"$python_binary" scripts/install_models.py --sam=False --metrics=True --runtime=True
"$python_binary" -m pytest tests/test_baselines.py -q > "$task_root/logs/$method-tests.log" 2>&1
"$python_binary" -m video3d.baselines evaluate --config="$task_root/config.json" --method="$method" \
  --output="$task_root/results/$method" --split=eval --limit=250 --k=4 \
  --chunk_size=2 --topk=4 --full_mesh=True --resume=True
TASK_ROOT="$task_root" METHOD="$method" "$python_binary" - <<'PY'
import json
import os
from pathlib import Path
root = Path(os.environ['TASK_ROOT'])
method = os.environ['METHOD']
rows = [json.loads(line) for line in (root/'results'/method/'evaluation.jsonl').read_text().splitlines()]
assert len(rows) == 250 and len({r['sample_id'] for r in rows}) == 250
required = ['viou','viou_aligned','chamfer','fscore@0.02','fscore@0.05','empty_pred','mesh_surface',
    'canonical/reference_visible/completeness_distance', 'canonical/reference_visible/recall_002']
for row in rows:
    assert all(key in row for key in required), row['sample_id']
    assert row['baseline']['scope'] == 'refined_mesh'
(root/'results'/method/'complete.json').write_text(json.dumps({'objects':250,'validated':True}))
PY
