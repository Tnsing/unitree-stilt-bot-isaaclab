# Execution plan

## Phase 0 — reproducible G1 warm-up

- [x] Locate local Isaac Lab and Unitree G1 asset support.
- [x] Run the native `Isaac-Velocity-Flat-G1-v0` smoke training.
- [x] Register a project-owned G1 warm-up task.
- [ ] Run the project-owned warm-up task and save its raw output.

## Phase 1 — attached-stilt asset

- [ ] Select G1 variant and exact joint/foot body names.
- [ ] Create or import a licensed stilt adapter USD.
- [ ] Validate mass, COM, friction, collision geometry and ankle/foot connection.
- [ ] Add contact sensors and deterministic reset tests.
- [ ] Run 1-environment visual and 8-environment contact smoke tests.

## Phase 2 — learning

- [ ] B1 no-imitation attached-stilt PPO.
- [ ] B2 foothold curriculum / terrain perception.
- [ ] B3 imitation warm-start using licensed retargeted motion.
- [ ] Three seeds, fixed evaluation suite, and checkpoint hashes.

## Phase 3 — evidence package

- [ ] Raw TensorBoard/scalars and config snapshots.
- [ ] Rollout videos.
- [ ] Failure taxonomy: slip, tip, collision, joint limit, command miss.
- [ ] Reproducibility report with exact Isaac Sim / Isaac Lab / Python / Torch versions.
