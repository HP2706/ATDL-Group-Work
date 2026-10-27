#!/usr/bin/env bash
# Run T6 inference on the VM after setup_vm.sh, from the root of this repo.
# Usage: bash scripts/t6/run_vm.sh            -> t6_outputs.tar.gz in the repo root
set -euo pipefail

SAM3D_DIR=${SAM3D_DIR:-$HOME/sam-3d-objects}
CONDA_DIR=${CONDA_DIR:-$HOME/miniforge3}
source "$CONDA_DIR/etc/profile.d/conda.sh"
conda activate sam3d-objects

python scripts/t6/run_sam3d.py --sam3d-dir "$SAM3D_DIR" --inputs t6/inputs --out t6/outputs --smoke \
    2>&1 | tee t6_smoke.log
python scripts/t6/run_sam3d.py --sam3d-dir "$SAM3D_DIR" --inputs t6/inputs --out t6/outputs \
    2>&1 | tee t6_run.log
mv t6_smoke.log t6_run.log t6/outputs/
tar czf t6_outputs.tar.gz t6/outputs
grep -c '"status": "ok"' t6/outputs/run_log.jsonl | xargs echo "ok runs:"
grep -c '"status": "error"' t6/outputs/run_log.jsonl | xargs echo "failed runs:" || true
