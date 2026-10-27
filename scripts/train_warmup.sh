#!/usr/bin/env bash
set -euo pipefail
export TERM=${TERM:-xterm}

ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
ISAACLAB_PATH=${ISAACLAB_PATH:-/data/robot_motion_lib_isaaclab_eval/IsaacLab}
ISAACLAB_PYTHON=${ISAACLAB_PYTHON:-/home/ninesoo/anaconda3/envs/isaacSim/bin/python}
WARP_PATH=${WARP_PATH:-/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64}
NUM_ENVS=64
MAX_ITERATIONS=10

while [[ $# -gt 0 ]]; do
  case "$1" in
    --num_envs) NUM_ENVS=$2; shift 2 ;;
    --max_iterations) MAX_ITERATIONS=$2; shift 2 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

export PYTHONPATH="$WARP_PATH:$ROOT/src:${PYTHONPATH:-}"
cd "$ROOT"
exec "$ISAACLAB_PATH/isaaclab.sh" -p "$ROOT/scripts/train_registered_task.py" \
  --task StiltBot-G1-Warmup-v0 \
  --num_envs "$NUM_ENVS" \
  --max_iterations "$MAX_ITERATIONS" \
  --headless \
  --experiment_name stiltbot_warmup
