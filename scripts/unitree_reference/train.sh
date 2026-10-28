#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "$0")/runtime.sh"
SKILL="${SKILL:?Set SKILL=walk/run/dance/stilt}"
case "$SKILL" in walk) NAME=Walk;;run) NAME=Run;;dance) NAME=Dance;;stilt) NAME=Stilt;;*) exit 2;;esac
exec python "$REFERENCE_ROOT/scripts/train_registered_task.py" --task "UnitreeRef-G1-29dof-${NAME}-v0" --num_envs "${NUM_ENVS:-4096}" --max_iterations "${MAX_ITERATIONS:-2000}" --seed "${SEED:-42}" --run_name "${RUN_NAME:-formal_unitree_ref}" --headless "$@"
