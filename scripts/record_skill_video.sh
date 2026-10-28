#!/usr/bin/env bash
set -euo pipefail
export TERM="${TERM:-xterm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-/data/robot_motion_lib_isaaclab_eval/IsaacLab}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/ninesoo/anaconda3/envs/isaacSim/bin/python}"
WARP_PATH="${WARP_PATH:-/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64}"
SKILL="${SKILL:-walk}"
RUN="${RUN:?Set RUN to the timestamped training run folder}"
CHECKPOINT="${CHECKPOINT:-model_1499.pt}"
VIDEO_LENGTH="${VIDEO_LENGTH:-500}"
NUM_ENVS="${NUM_ENVS:-1}"
if [[ -x "${ISAACLAB_PYTHON}" && -f "/home/ninesoo/anaconda3/etc/profile.d/conda.sh" ]]; then
  source /home/ninesoo/anaconda3/etc/profile.d/conda.sh
  conda activate "$(dirname "$(dirname "${ISAACLAB_PYTHON}")")"
fi
if [[ -n "${TASK_ID:-}" && -n "${EXPERIMENT_NAME:-}" ]]; then
  TASK="${TASK_ID}"; EXPERIMENT="${EXPERIMENT_NAME}"
else
  case "${SKILL}" in
    walk) TASK="StiltBot-G1-Walk-v0"; EXPERIMENT="stiltbot_g1_walk_formal" ;;
    run) TASK="StiltBot-G1-Run-v0"; EXPERIMENT="stiltbot_g1_run_formal" ;;
    jump) TASK="StiltBot-G1-Jump-v0"; EXPERIMENT="stiltbot_g1_jump_formal" ;;
    dance) TASK="StiltBot-G1-Dance-v0"; EXPERIMENT="stiltbot_g1_dance_formal" ;;
    *) echo "Unsupported SKILL=${SKILL}; set TASK_ID and EXPERIMENT_NAME for custom tasks" >&2; exit 2 ;;
  esac
fi
if [[ "${CHECKPOINT}" = /* ]]; then
  CHECKPOINT_PATH="${CHECKPOINT}"
else
  CHECKPOINT_PATH="${ROOT}/logs/rsl_rl/${EXPERIMENT}/${RUN}/${CHECKPOINT}"
fi
export PYTHONPATH="${WARP_PATH}:${ROOT}/src:${PYTHONPATH:-}"
cd "${ROOT}"
exec "${ISAACLAB_PATH}/isaaclab.sh" -p "${ROOT}/scripts/play_registered_task.py" \
  --task "${TASK}" --num_envs "${NUM_ENVS}" --video --video_length "${VIDEO_LENGTH}" \
  --experiment_name "${EXPERIMENT}" --load_run "${RUN}" --checkpoint "${CHECKPOINT_PATH}" "$@"
