#!/usr/bin/env python3
"""Use versioned pre-autoreset audit; never modify historical audit outputs.

The original evaluator had a stale dance target cache after full reset and read
robot state after autoreset. V2 preserves first-step termination checks and captures
state at the termination computation boundary. See docs/beyondmimic_reproduction.md.
"""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parents[1]/'beyondmimic/evaluate.py'),run_name='__main__')
