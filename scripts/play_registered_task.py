#!/usr/bin/env python3
"""Import project task registrations and run the project-owned play implementation."""
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path

import stilt_bot  # noqa: F401  (registers Gym tasks)

isaaclab_root = Path(os.environ.get("ISAACLAB_PATH", "/data/robot_motion_lib_isaaclab_eval/IsaacLab")).resolve()
play_script = Path(__file__).resolve().with_name("play_task_impl.py")
sys.path.insert(0, str(isaaclab_root / "scripts/reinforcement_learning/rsl_rl"))
if not play_script.exists():
    raise FileNotFoundError(f"Missing project play script: {play_script}")
runpy.run_path(str(play_script), run_name="__main__")
