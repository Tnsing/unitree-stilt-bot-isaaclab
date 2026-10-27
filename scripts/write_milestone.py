#!/usr/bin/env python3
"""Write an auditable milestone manifest for one completed G1 skill run."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def last_scalars(event_files: list[Path]) -> dict[str, float | int]:
    try:
        from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    except Exception:
        return {}
    values: dict[str, tuple[int, float]] = {}
    for event_file in event_files:
        acc = EventAccumulator(str(event_file), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            if events:
                e = events[-1]
                values[tag] = (int(e.step), float(e.value))
    return {tag: {"step": step, "value": value} for tag, (step, value) in sorted(values.items())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skill", required=True)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--video", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--experiment", required=True)
    ap.add_argument("--num-envs", type=int, required=True)
    ap.add_argument("--max-iterations", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--train-log", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir).resolve()
    video = Path(args.video).resolve()
    out = Path(args.out).resolve()
    checkpoints = sorted(run_dir.glob("model_*.pt"), key=lambda p: int(p.stem.split("_")[-1]))
    events = sorted(run_dir.glob("events.out.tfevents.*"))
    env_yaml = run_dir / "params" / "env.yaml"
    agent_yaml = run_dir / "params" / "agent.yaml"
    train_log = Path(args.train_log).resolve()
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)
    if not checkpoints:
        raise FileNotFoundError(f"No checkpoints in {run_dir}")
    if not video.is_file() or video.stat().st_size < 1024:
        raise RuntimeError(f"Video missing or suspiciously small: {video}")
    if not env_yaml.is_file() or not agent_yaml.is_file():
        raise FileNotFoundError("Missing Isaac Lab config snapshots")

    git_rev = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    manifest = {
        "schema": "stiltbot.formal-milestone.v1",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "skill": args.skill,
        "task": args.task,
        "experiment": args.experiment,
        "training": {
            "status": "completed",
            "num_envs": args.num_envs,
            "max_iterations": args.max_iterations,
            "seed": args.seed,
            "run_dir": str(run_dir),
            "train_log": str(train_log),
            "config_env": str(env_yaml.resolve()),
            "config_agent": str(agent_yaml.resolve()),
            "event_files": [str(p.resolve()) for p in events],
            "checkpoints": [
                {"path": str(p.resolve()), "iteration": int(p.stem.split("_")[-1]), "sha256": sha256(p)}
                for p in checkpoints
            ],
        },
        "demonstration": {
            "status": "recorded",
            "video": str(video),
            "video_bytes": video.stat().st_size,
            "video_sha256": sha256(video),
        },
        "metrics_last": last_scalars(events),
        "provenance": {
            "git_revision": git_rev,
            "repo": str(Path.cwd().resolve()),
            "stilt_geometry_registered": False,
            "asset_scope": "native Unitree G1 minimal asset on flat ground",
        },
    }
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"out": str(out), "final_checkpoint": str(checkpoints[-1]), "video": str(video)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
