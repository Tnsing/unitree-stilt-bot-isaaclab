#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
source /home/ninesoo/anaconda3/etc/profile.d/conda.sh
conda activate isaacSim
export PYTHONPATH="/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64:$ROOT/src:$ROOT/scripts:${PYTHONPATH:-}"
export TERM=xterm PYTHONUNBUFFERED=1
exec /home/ninesoo/anaconda3/envs/isaacSim/bin/python "$ROOT/scripts/evaluate_skill.py" "$@" --headless
