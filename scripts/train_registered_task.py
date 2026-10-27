#!/usr/bin/env python3
"""Import project task registrations, then delegate to Isaac Lab's RSL-RL trainer."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import stilt_bot  # noqa: F401,E402  (registers Gym tasks)

isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", "/data/robot_motion_lib_isaaclab_eval/IsaacLab")).resolve()
train_script = isaaclab_root / "scripts/reinforcement_learning/rsl_rl/train.py"
sys.path.insert(0, str(train_script.parent))
if not train_script.exists():
    raise FileNotFoundError(f"Set ISAACLAB_PATH and call scripts/train_warmup.sh; missing {train_script}")
runpy.run_path(str(train_script), run_name="__main__")
