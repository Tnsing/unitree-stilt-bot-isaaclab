#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /home/ninesoo/anaconda3/etc/profile.d/conda.sh
conda activate isaacSim
export PYTHONPATH="/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64:$ROOT/src:${PYTHONPATH:-}"
export TERM=xterm PYTHONUNBUFFERED=1
SKILL="${SKILL:?Set SKILL to run, jump or dance}"
case "$SKILL" in
 run) TASK=StiltBot-G1-Run-v1;;
 jump) TASK=StiltBot-G1-Jump-v1;;
 dance) TASK=StiltBot-G1-Dance-v1;;
 *) exit 2;;
esac
cd "$ROOT"
exec /home/ninesoo/anaconda3/envs/isaacSim/bin/python "$ROOT/scripts/train_registered_task.py" --task "$TASK" --num_envs "${NUM_ENVS:-4096}" --max_iterations "${MAX_ITERATIONS:-1500}" --seed "${SEED:-42}" --run_name "${RUN_NAME:-formal_phase_v1_seed42}" --headless "$@"
