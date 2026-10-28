# Formal training evidence (2026-10-28)

This index records the local evidence produced by the formal Isaac Lab runs. The experiments use the native Unitree G1 minimal USD on a flat plane; no stilt USD, additional rigid body, or constraint is registered.

All formal training runs use 4,096 environments, seed 42, and 1,500 PPO iterations. The independent audits use 64 held-out simulator environments, a different evaluation seed (`20261027`), and 1,000 steps. The audit is simulator-only evidence and is not hardware validation.

## Final evidence mapping

| Skill | Task | Training | Milestone | Checkpoint | Demo video | Audit |
|---|---|---|---|---|---|---|
| Walk | `StiltBot-G1-Walk-v0` | completed | `artifacts/formal/walk.json` | `logs/rsl_rl/stiltbot_g1_walk_formal/2026-10-27_21-37-38_formal_seed42/model_1499.pt` | `logs/rsl_rl/stiltbot_g1_walk_formal/2026-10-27_21-37-38_formal_seed42/videos/play/rl-video-episode-0.mp4` | `artifacts/audit/walk_v0_64/evaluation.json` — **passed** |
| Run | `StiltBot-G1-Run-v1` | completed | `artifacts/formal/run-v1.json` | `logs/rsl_rl/stiltbot_g1_run_v1/2026-10-28_09-08-20_formal_phase_v1_seed42/model_1499.pt` | `logs/rsl_rl/stiltbot_g1_run_v1/2026-10-28_09-08-20_formal_phase_v1_seed42/videos/play/rl-video-episode-0.mp4` | `artifacts/audit/run_v1_64/evaluation.json` — **passed** |
| Jump | `StiltBot-G1-Jump-v1` | completed | `artifacts/formal/jump-v1.json` | `logs/rsl_rl/stiltbot_g1_jump_v1/2026-10-28_09-38-19_formal_phase_v1_seed42/model_1499.pt` | `logs/rsl_rl/stiltbot_g1_jump_v1/2026-10-28_09-38-19_formal_phase_v1_seed42/videos/play/rl-video-episode-0.mp4` | `artifacts/audit/jump_v1_64/evaluation.json` — **passed** |
| Dance | `StiltBot-G1-Dance-v1` | completed | `artifacts/formal/dance-v1.json` | `logs/rsl_rl/stiltbot_g1_dance_v1/2026-10-28_10-09-53_formal_phase_v1_seed42/model_1499.pt` | `logs/rsl_rl/stiltbot_g1_dance_v1/2026-10-28_10-09-53_formal_phase_v1_seed42/videos/play/rl-video-episode-0.mp4` | `artifacts/audit/dance_v1_64/evaluation.json` — **not passed** |

The formal dance run is complete and has a checkpoint and a valid 10.02-second, 501-frame video. Its independent audit exposed a stability problem: 50% physical fall rate and 50% full-horizon survival. The phase-aligned body/yaw and arm errors were within their numerical thresholds, but the survival gate correctly failed. This is retained as a negative result rather than being reported as a successful dance controller.

## Video and hash checks

The recorded demonstrations are 1280×720, 50 FPS, 501 frames, and 10.02 seconds. Each milestone JSON records the SHA-256 of the final checkpoint and the demonstration video, along with the TensorBoard event file, environment/agent YAML snapshots, and periodic checkpoint hashes.

## Interpretation boundary

- `v0` walk is the native velocity-tracking baseline.
- `v1` run adds an observable phase clock and contact/flight terms.
- `v1` jump adds observable phase, root-local state, contact state, and a synthetic ballistic height reference.
- `v1` dance adds observable phase and synthetic body/upper-limb wave targets.
- The jump and dance targets are reproducible synthetic phase references; they are not motion-capture imitation and do not reproduce BeyondMimic/MimicKit behavior yet.
- A passing simulator audit does not establish sim-to-real transfer, balance on hardware, actuator safety, or real-robot success.
