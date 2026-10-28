#!/usr/bin/env bash
set -euo pipefail
REFERENCE_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
source /home/ninesoo/anaconda3/etc/profile.d/conda.sh
conda activate isaacSim
export ISAACLAB_PATH="${ISAACLAB_PATH:-/data/robot_motion_lib_isaaclab_eval/IsaacLab}"
export PYTHONPATH="$REFERENCE_ROOT/vendor:$REFERENCE_ROOT/src:/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64:${PYTHONPATH:-}"
export TERM=xterm PYTHONUNBUFFERED=1
export UNITREE_REFERENCE_ROOT="$REFERENCE_ROOT"
cd "$REFERENCE_ROOT"
