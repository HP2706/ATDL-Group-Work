#!/usr/bin/env bash
# Eight provider-native CPU jobs share an immutable archive ownership plan.
set -euo pipefail
task_root="${1:?persistent workspace required}"
plan="${2:?immutable plan required}"
shard="${3:?shard index required}"
code_root="$task_root/repo/Assignment 3 - Group/code"
printf -v shard_name 'shard-%02d' "$shard"
state_root="$(dirname "$plan")/$shard_name"
mkdir -p "$state_root"
exec > >(tee -a "$state_root/job.log") 2>&1
finish() {
  status=$?
  printf '{"exit_code":%d,"finished_utc":"%s"}\n' "$status" "$(date -u +%FT%TZ)" > "$state_root/exit.json"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LP_NUM_THREADS=2
sudo apt-get update -qq
sudo apt-get install -y -qq aria2 > "$state_root/system.log" 2>&1
python_binary="$task_root/runtime/cpu-venv/bin/python"
cd "$code_root"
git rev-parse HEAD > "$state_root/source-commit.txt"
downloaded=0
for attempt in 1 2 3; do
  if "$python_binary" scripts/download_omniobject.py --out "$task_root/data/raw/omniobject3d" --workers 4 --plan "$plan" --shard "$shard"; then
    downloaded=1
    break
  fi
  echo "Resuming failed archives, attempt $attempt"
done
test "$downloaded" = 1
printf '{"complete":true,"completed_utc":"%s"}\n' "$(date -u +%FT%TZ)" > "$state_root/download-complete.json"
if [[ "$shard" == 0 ]]; then
  PLAN_PATH="$plan" "$python_binary" - <<'PY'
import json,os,time
from pathlib import Path
plan = Path(os.environ['PLAN_PATH'])
groups = json.loads(plan.read_text())['shards']
deadline = time.monotonic() + 47*3600
while time.monotonic() < deadline:
    directories = [plan.parent/f'shard-{i:02d}' for i in range(len(groups))]
    for directory in directories:
        exit_file = directory/'exit.json'
        if exit_file.exists() and json.loads(exit_file.read_text())['exit_code']:
            raise RuntimeError(f'Download worker failed: {directory}')
    if all((directory/'download-complete.json').exists() for directory in directories):
        break
    time.sleep(30)
else:
    raise TimeoutError('Other archive shards did not finish within the batch budget')
print('All archive shards verified and extracted; starting full preparation', flush=True)
PY
  # Reuse the maintained complete preparation/check pipeline. Its download stage
  # now finds every archive complete and reuses them without transferring again.
  bash ucloud/prepare-scanned.sh "$task_root"
fi
