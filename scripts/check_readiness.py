#!/usr/bin/env python3
"""Audit the local Isaac Lab / Unitree G1 prerequisites without launching Kit."""
from __future__ import annotations

import argparse
import importlib.metadata as md
import json
import shutil
import subprocess
from pathlib import Path


def version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--isaaclab", type=Path, required=True)
    p.add_argument("--python", type=Path, required=True)
    args = p.parse_args()

    isaaclab = args.isaaclab.resolve()
    checks: dict[str, object] = {
        "isaaclab_exists": isaaclab.is_dir(),
        "isaaclab_version_file": (isaaclab / "VERSION").read_text().strip() if (isaaclab / "VERSION").exists() else None,
        "python_exists": args.python.exists(),
        "gpu": None,
        "packages": {name: version(name) for name in ("isaacsim", "isaaclab", "isaaclab_tasks", "rsl-rl-lib", "torch", "gymnasium")},
        "g1_config": (isaaclab / "source/isaaclab_assets/isaaclab_assets/robots/unitree.py").exists(),
        "g1_task": (isaaclab / "source/isaaclab_tasks/isaaclab_tasks/manager_based/locomotion/velocity/config/g1").is_dir(),
    }
    if shutil.which("nvidia-smi"):
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,memory.free", "--format=csv,noheader"], text=True)
            checks["gpu"] = [line.strip() for line in out.splitlines() if line.strip()]
        except subprocess.CalledProcessError as exc:
            checks["gpu_error"] = str(exc)
    print(json.dumps(checks, indent=2, ensure_ascii=False))
    required = ("isaaclab_exists", "python_exists", "g1_config", "g1_task")
    return 0 if all(checks[k] for k in required) else 1


if __name__ == "__main__":
    raise SystemExit(main())
