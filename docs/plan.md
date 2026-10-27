# Execution plan

## Phase 0 — runtime and native G1 baseline

- [x] Locate local Isaac Lab and Unitree G1 asset support.
- [x] Run the native `Isaac-Velocity-Flat-G1-v0` smoke training.
- [x] Register a project-owned G1 warm-up task.
- [x] Run the project-owned warm-up task and preserve its raw output.

## Phase 1 — four non-stilt skills

- [x] Register separate Unitree G1 task IDs for walk, run, jump and dance.
- [x] Keep the scene on a flat plane and verify no stilt geometry is registered.
- [x] Validate all four task configurations with a 4-environment, 2-iteration Isaac Lab smoke run.
- [x] Add a checkpoint-based video recorder and an auditable milestone writer.
- [ ] Complete formal 4096-environment, 1500-iteration training with seed 42 for each skill.
- [ ] Save every periodic checkpoint, TensorBoard events, env/agent YAML snapshots and exact stdout logs.
- [ ] Record one 500-step demonstration video from the final checkpoint for each skill.
- [ ] Verify each milestone manifest against files, hashes, video size and TensorBoard scalars.

## Phase 2 — evidence package

- [ ] Generate a final four-skill evidence index with training status and last metrics.
- [ ] Review videos and label them as simulator demonstrations, not real-robot validation.
- [ ] Report the explicit boundary that jump/dance use reproducible synthetic phase references rather than motion-capture imitation.
- [ ] Keep the attached-stilt phase separate until a licensed/validated USD and contact protocol exist.

## Deferred — attached stilts

- [ ] Select G1 variant and exact joint/foot body names.
- [ ] Create or import a licensed stilt adapter USD.
- [ ] Validate mass, COM, friction, collision geometry and ankle/foot connection.
- [ ] Add contact sensors and deterministic reset tests.
- [ ] Run 1-environment visual and 8-environment contact smoke tests before any stilt training.
