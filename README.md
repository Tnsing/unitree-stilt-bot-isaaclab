# Unitree Stilt-Bot Isaac Lab

基于宇树 Unitree G1 人形机器人、Isaac Lab 和 RSL-RL PPO 的“踩高跷”研究项目。

This repository is the experiment ledger and implementation scaffold for training a Unitree humanoid to walk on stilts. It deliberately separates:

1. **Research evidence**: BeyondMimic, MimicKit, BeamDojo and the Stilt-bot design direction.
2. **G1 warm-up control**: a reproducible Isaac Lab baseline on the Unitree G1.
3. **Stilt dynamics**: the next stage, which must use an attached stilt adapter asset or a validated articulated stilt model. A flat-ground G1 result is not reported as stilt walking.

## Current status (2026-10-27)

- Research notes and implementation decisions: [`docs/research.md`](docs/research.md).
- Readiness audit: [`scripts/check_readiness.py`](scripts/check_readiness.py).
- Registered Isaac Lab warm-up task: `StiltBot-G1-Warmup-v0`.
- G1 baseline smoke run completed on the local RTX 4090 with Isaac Lab 2.3.0 / Isaac Sim 5.1.0: 64 environments, 10 PPO iterations, 15,360 environment steps. This is software/runtime evidence only; it is not a claim of learned walking or stilt walking.
- The next controlled experiment is the attached-stilt task, with contact/geometry validation before long training.

## Quick start

The commands below use the local Isaac Lab checkout and conda environment. Do not copy Isaac Lab into this repository; keep it as an external dependency.

```bash
export ISAACLAB_PATH=/data/robot_motion_lib_isaaclab_eval/IsaacLab
export ISAACLAB_PYTHON=/home/ninesoo/anaconda3/envs/isaacSim/bin/python
export WARP_PATH=/home/ninesoo/wzh/isaac-sim/isaac-sim-standalone-5.1.0-linux-x86_64/extscache/omni.warp.core-1.8.2+lx64
export PYTHONPATH="$WARP_PATH:$PWD/src:${PYTHONPATH:-}"

python scripts/check_readiness.py \
  --isaaclab "$ISAACLAB_PATH" \
  --python "$ISAACLAB_PYTHON"

bash scripts/train_warmup.sh --num_envs 64 --max_iterations 10
```

For a real experiment, increase the budget only after the task-level checks pass. Suggested first controlled run: 4096 environments, 2,000 iterations, three seeds, and the same policy/physics settings for all ablations.

## Scope boundaries

- Unitree assets remain external and are not redistributed here. Follow the Unitree asset license and Isaac Lab asset terms.
- Human motion data and retargeted motion data must retain their original licenses and provenance.
- `StiltBot-G1-Warmup-v0` is a warm-up task. It is not the final attached-stilt physics task.
- A successful simulator process, an increasing reward, or a checkpoint file is not sufficient evidence of stable stilt walking. The final report must include episode success, fall rate, support/contact diagnostics, multi-seed results, and a rollout video.
