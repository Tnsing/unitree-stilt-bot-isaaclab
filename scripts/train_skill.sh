#!/usr/bin/env bash
set -euo pipefail
export TERM="${TERM:-xterm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-/data/robot_motion_lib_isaaclab_eval/IsaacLab}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/ninesoo/anaconda3/envs/isaacSim/bin/python}"
WARP_PATH="${WARP_PATH:-/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64}"
SKILL="${SKILL:-walk}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
SEED="${SEED:-42}"
RUN_NAME="${RUN_NAME:-formal_seed${SEED}}"
if [[ -x "${ISAACLAB_PYTHON}" && -f "/home/ninesoo/anaconda3/etc/profile.d/conda.sh" ]]; then
  source /home/ninesoo/anaconda3/etc/profile.d/conda.sh
  conda activate "$(dirname "$(dirname "${ISAACLAB_PYTHON}")")"
fi
case "${SKILL}" in
  walk) TASK="StiltBot-G1-Walk-v0"; EXPERIMENT="stiltbot_g1_walk_formal" ;;
  run) TASK="StiltBot-G1-Run-v0"; EXPERIMENT="stiltbot_g1_run_formal" ;;
  jump) TASK="StiltBot-G1-Jump-v0"; EXPERIMENT="stiltbot_g1_jump_formal" ;;
  dance) TASK="StiltBot-G1-Dance-v0"; EXPERIMENT="stiltbot_g1_dance_formal" ;;
  *) echo "Unsupported SKILL=${SKILL}; use walk, run, jump or dance" >&2; exit 2 ;;
esac
export PYTHONPATH="${WARP_PATH}:${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${ROOT}/scripts/train_registered_task.py" \
  --task "${TASK}" --num_envs "${NUM_ENVS}" --max_iterations "${MAX_ITERATIONS}" \
  --seed "${SEED}" --run_name "${RUN_NAME}" --experiment_name "${EXPERIMENT}" "$@"
