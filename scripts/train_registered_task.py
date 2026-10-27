#!/usr/bin/env python3
"""Import project task registrations, then delegate to Isaac Lab's RSL-RL trainer."""
from __future__ import annotations

import runpy
from pathlib import Path

import stilt_bot  # noqa: F401,E402  (registers Gym tasks)

isaaclab_root = Path(__file__).resolve().parents[2] / "robot_motion_lib_isaaclab_eval" / "IsaacLab"
train_script = isaaclab_root / "scripts/reinforcement_learning/rsl_rl/train.py"
if not train_script.exists():
    raise FileNotFoundError(f"Set ISAACLAB_PATH and call scripts/train_warmup.sh; missing {train_script}")
runpy.run_path(str(train_script), run_name="__main__")
