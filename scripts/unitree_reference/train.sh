#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
SKILL="${SKILL:?Set SKILL=walk/run/dance/stilt}"
case "$SKILL" in walk) NAME=Walk;;run) NAME=Run;;dance) NAME=Dance;;stilt) NAME=Stilt;;*) exit 2;;esac
python - <<'PYTHON'
from datetime import datetime, timezone
import hashlib,json,os,subprocess
from pathlib import Path
root=Path(os.environ['UNITREE_REFERENCE_ROOT'])
folder=root/'logs/unitree_reference';folder.mkdir(parents=True,exist_ok=True)
record={'created_at_utc':datetime.now(timezone.utc).isoformat(),'skill':os.environ['SKILL'],'git_revision_at_launch':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),'git_dirty_at_launch':bool(subprocess.check_output(['git','status','--porcelain'],text=True).strip()),'source_sha256':{str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['src/stilt_bot/tasks/unitree_reference','vendor/unitree_rl_lab','scripts/unitree_reference'] for p in (root/folder).rglob('*.py')},'runtime':'Lab 2.3 / Sim 5.1 / PhysX'}
(folder/(os.environ['SKILL']+'.launch.json')).write_text(json.dumps(record,indent=2))
PYTHON
exec python "$REFERENCE_ROOT/scripts/train_registered_task.py" --task "UnitreeRef-G1-29dof-${NAME}-v0" --num_envs "${NUM_ENVS:-4096}" --max_iterations "${MAX_ITERATIONS:-2000}" --seed "${SEED:-42}" --run_name "${RUN_NAME:-formal_unitree_ref}" --headless "$@"
