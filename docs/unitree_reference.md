# Unitree RL Lab reference training

The requested reference is `unitreerobotics/unitree_rl_lab`, branch `isaaclab3.0`, pinned at commit `2522f89ab8ad6b1188b0368d122d16cb0e3cb8c8`. The upstream README specifies Isaac Sim 6.0 and Isaac Lab 3.0. This machine has Isaac Sim 5.1 / Isaac Lab 2.3, so the selected manager-based PhysX tasks are explicitly backported. This is not a native Lab 3.0 or Newton training run.

## What is reused

The official G1 29 DOF URDF, actuator gains and joint names replace the earlier Isaac Lab 37-joint minimal G1. The velocity tasks reuse upstream history observations, command tracking, gait/contact rewards and a 512/256/128 actor/critic. Dance uses the upstream `dance_102` CSV and MotionCommand imitation, with anchor/body positions, orientations and velocities as tracking objectives. It does not reuse the earlier synthetic arm-wave policy.

The local extensions define separate walk and run command distributions, observable run phase and flight rewards, close-up policy videos, contact auditing, and a parameterized stilt prototype. New Gym IDs are `UnitreeRef-G1-29dof-Walk-v0`, `UnitreeRef-G1-29dof-Run-v0`, `UnitreeRef-G1-29dof-Dance-v0`, and `UnitreeRef-G1-29dof-Stilt-v0`.

## Compatibility changes

`prepare_compat.py` keeps the original snapshot unchanged and writes an adapted package under `vendor/`. Its unified diff is `artifacts/upstream/unitree_isaaclab23.patch`.

- PhysX and contact-sensor presets are resolved to the Lab 2.3 PhysX configuration; Newton/direct/Warp tasks are not registered.
- Torch tensor access replaces the 3.0 Warp-view wrapper at the API boundary.
- Lab 2.3 articulation write and center-of-mass APIs are used.
- Dance CSV root quaternions are converted from XYZW to WXYZ. MotionCommand identity quaternions are WXYZ.
- RSL-RL observation routing is explicitly `policy -> policy`, `critic -> critic`, with observation normalization stated in the config. This uses an asymmetric critic and is not a controlled reproduction of the upstream symmetric-critic configuration.

## Stilt geometry and evidence boundary

The prototype adds one rigid body to each ankle using a fixed joint. Each stilt contains a 0.30 m cylindrical pole and a 0.12 x 0.08 x 0.02 m pad, with 0.8 kg mass per side and analytic combined inertia. There are no added actuators, kinematic supports, floating-base locks, or external stabilizing forces. Imported fixed links remain separate so their ground-contact forces are observable. This geometry is an explicit simulation design, not a validated model of real stilt hardware.

`validate_stilt.py` checks 29 actuated joints, two stilt bodies, their imported masses/inertias, two fixed joints, four collision shapes and actual tip contact. It records that default PD alone tips over; passing the import/contact check is not passing the locomotion gate. The current trace has negligible penetration and no original-ankle ground contact.

## Reproduce

```bash
cd /data/踩高跷
# Set HTTPS_PROXY if required by this host's network.
python scripts/unitree_reference/fetch_sources.py
python scripts/unitree_reference/prepare_compat.py
python scripts/unitree_reference/build_stilt_urdf.py
source scripts/unitree_reference/runtime.sh
python scripts/unitree_reference/convert_dance.py --headless
python scripts/unitree_reference/validate_stilt.py --headless
SKILL=walk NUM_ENVS=4 MAX_ITERATIONS=2 RUN_NAME=compat_smoke scripts/unitree_reference/train.sh
# A fresh serial suite, without adoption of a running process:
python scripts/unitree_reference/run_suite.py --iterations 2000 --num-envs 4096
```

The source/asset payloads are SHA-256 locked in `configs/unitree_upstream_lock.json` and `configs/unitree_asset_lock.json`. Downloaded robot meshes and generated USD/NPZ files remain local. Original source attribution/license headers are retained. The original Unitree source/asset licenses continue to apply; the project's MIT license does not relicense third-party content or motion data.

## Training and evaluation

Each task starts with a 2,000-iteration, 4,096-environment, seed-42 first-stage training budget. Upstream defaults are much longer (50,000 locomotion iterations and 30,000 mimic iterations). Completion of this budget does not imply convergence. Models are saved every 100 iterations; checkpoint-500, checkpoint-1000 and final policy videos are recorded. Videos are actual policy inference in physics, and are distinguished from the reference FK conversion.

The independent final audit uses 64 environments, seed 20261028, no observation noise and no pushes; startup mass/material/COM randomization remains where configured. Walk/run/stilt use fixed held-out commands and 30 seconds. Dance starts at reference frame zero and covers the complete clip without counting reference-boundary resets as survival. Short milestone videos cannot pass the full-horizon gate. Dead and reset-state samples are excluded.

Walk requires >=95% survival and speed RMSE <=0.25 m/s. Run uses <=0.4 m/s and requires repeated force-backed flight/landing. Dance requires >=95% full-clip survival, joint RMSE <=0.3 rad and body-position mean error <=0.15 m. Stilt walking additionally requires >=0.5 m forward displacement in >=80% environments with actual stilt support. Passing any of these nominal tests is not evidence of broad robustness or hardware transfer.

Progress is stored in `artifacts/unitree_reference/progress.json`. Per-skill `milestone.json` separates `training_completed` from `task_gate_passed`, and includes configs, event/checkpoint/video hashes and audit paths. Raw stdout is under `logs/unitree_reference/`; all generated demonstrations are under `artifacts/unitree_reference/<skill>/video_<iteration>/demo.mp4`.
