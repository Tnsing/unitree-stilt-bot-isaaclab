#!/usr/bin/env bash
set -euo pipefail
export TERM="${TERM:-xterm}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ISAACLAB_PATH="${ISAACLAB_PATH:-/data/robot_motion_lib_isaaclab_eval/IsaacLab}"
ISAACLAB_PYTHON="${ISAACLAB_PYTHON:-/home/ninesoo/anaconda3/envs/isaacSim/bin/python}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITERATIONS="${MAX_ITERATIONS:-1500}"
SEED="${SEED:-42}"
FORMAL_RUN_NAME="${FORMAL_RUN_NAME:-formal_seed${SEED}}"
SUITE_LOG_DIR="${ROOT}/logs/formal_suite"
mkdir -p "${SUITE_LOG_DIR}" "${ROOT}/artifacts/formal"
exec 9>"${SUITE_LOG_DIR}/suite.lock"
flock -n 9 || { echo "Another formal suite is already running" >&2; exit 3; }

for SKILL in walk run jump dance; do
  case "${SKILL}" in
    walk) TASK="StiltBot-G1-Walk-v0"; EXPERIMENT="stiltbot_g1_walk_formal" ;;
    run) TASK="StiltBot-G1-Run-v0"; EXPERIMENT="stiltbot_g1_run_formal" ;;
    jump) TASK="StiltBot-G1-Jump-v0"; EXPERIMENT="stiltbot_g1_jump_formal" ;;
    dance) TASK="StiltBot-G1-Dance-v0"; EXPERIMENT="stiltbot_g1_dance_formal" ;;
  esac
  TRAIN_LOG="${SUITE_LOG_DIR}/${SKILL}.train.log"
  echo "[$(date -Is)] START ${SKILL}" | tee -a "${SUITE_LOG_DIR}/suite.log"
  SKILL="${SKILL}" NUM_ENVS="${NUM_ENVS}" MAX_ITERATIONS="${MAX_ITERATIONS}" SEED="${SEED}" RUN_NAME="${FORMAL_RUN_NAME}" \
    "${ROOT}/scripts/train_skill.sh" --headless >"${TRAIN_LOG}" 2>&1
  RUN_DIR="$(find "${ROOT}/logs/rsl_rl/${EXPERIMENT}" -mindepth 1 -maxdepth 1 -type d -name "*_${FORMAL_RUN_NAME}" -printf '%T@ %p\n' | sort -n | tail -1 | cut -d' ' -f2-)"
  test -n "${RUN_DIR}" && test -d "${RUN_DIR}"
  FINAL_CHECKPOINT="$(find "${RUN_DIR}" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' | sort -V | tail -1)"
  test -n "${FINAL_CHECKPOINT}"
  SKILL="${SKILL}" RUN="$(basename "${RUN_DIR}")" CHECKPOINT="${FINAL_CHECKPOINT}" VIDEO_LENGTH="${VIDEO_LENGTH:-500}" \
    "${ROOT}/scripts/record_skill_video.sh" --headless >"${SUITE_LOG_DIR}/${SKILL}.video.log" 2>&1
  VIDEO="${RUN_DIR}/videos/play/rl-video-episode-0.mp4"
  "${ISAACLAB_PYTHON}" "${ROOT}/scripts/write_milestone.py" --skill "${SKILL}" --task "${TASK}" --experiment "${EXPERIMENT}" \
    --run-dir "${RUN_DIR}" --video "${VIDEO}" --num-envs "${NUM_ENVS}" --max-iterations "${MAX_ITERATIONS}" \
    --seed "${SEED}" --train-log "${TRAIN_LOG}" --out "${ROOT}/artifacts/formal/${SKILL}.json"
  echo "[$(date -Is)] DONE ${SKILL} ${RUN_DIR}" | tee -a "${SUITE_LOG_DIR}/suite.log"
done

echo "[$(date -Is)] COMPLETE all four skills" | tee -a "${SUITE_LOG_DIR}/suite.log"
