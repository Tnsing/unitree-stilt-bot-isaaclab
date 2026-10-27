# Related-work research and implementation decisions

Last checked: 2026-10-27 (Asia/Shanghai).

## User-provided references

| Reference | Role in this project | Evidence boundary |
|---|---|---|
| [BeyondMimic, arXiv:2508.08241](https://arxiv.org/abs/2508.08241) and [project page](https://beyondmimic.github.io/) | Motion tracking and shared hyperparameter direction for Unitree humanoids; useful for retargeted reference motion and sim-to-real discipline. | The original `HybridRobotics/whole_body_tracking` repository is now marked retired; its README points to `mjlab` for continued tracking work. We use the ideas and code structure as a reference, not as an unqualified dependency. |
| [MimicKit, arXiv:2510.13794](https://arxiv.org/abs/2510.13794) and [`xbpeng/MimicKit`](https://github.com/xbpeng/MimicKit) | A modular comparison frame for DeepMimic, AMP, AWR, ASE, LCP, ADD and SMP. | The framework is not Isaac-Lab-native in this project. It informs ablations and motion-imitation interfaces. |
| [BeamDojo, arXiv:2502.10363](https://arxiv.org/abs/2502.10363) and [project page](https://why618188.github.io/beamdojo/) | Sparse footholds, sampling-based foothold reward, double critic, two-stage training, and onboard elevation-map direction. | The project page is a research reference; no assumption is made that its implementation is directly reusable for G1 or stilts. |
| *A Simplified Underactuated Platform for AI-Ready Bipedal Walking Control: The Stilt-bot* | Hardware abstraction and underactuated walking-control direction for a simplified stilt platform. | Exact implementation details and repository availability must be re-verified before claiming a reproduction. |

## Synthesis

### 1. Motion imitation is a curriculum tool, not the stilt task itself

BeyondMimic and MimicKit suggest a practical order: first obtain a stable low-level controller and a motion/reference interface, then add task objectives. For this project the initial controller should be evaluated on the native G1 configuration before the stilt adapter is introduced. This keeps failures attributable to the stilt contact model rather than to an unvalidated humanoid baseline.

### 2. Sparse footholds require dense shaping and staged difficulty

BeamDojo's useful transferable ideas are: (a) a foothold-aware reward, (b) a dense locomotion critic or objective so that learning does not wait for rare successes, and (c) a flat-to-task-terrain curriculum. The stilt task will use the analogous phases:

1. flat G1 velocity warm-up;
2. attached stilts on a flat plane, with randomized friction/mass/COM;
3. narrow beams or sparse footholds;
4. speed, turns, stop-start, recovery and disturbance evaluation.

### 3. The key engineering risk is the stilt attachment model

A box spawned near a foot is not an attached stilt. Phase 1 must validate one of the following:

- a USD articulation with a fixed/hinged stilt adapter connected to the ankle/foot;
- a single combined robot-plus-stilts USD with correct inertial and collision properties;
- a physically validated constraint model whose contacts and wrench transfer are logged.

Kinematic “follow-the-foot” proxies may be used only for software smoke tests and must be labeled as such.

## Proposed MDP

**Policy inputs**

- G1 base linear/angular velocity and projected gravity;
- commanded planar velocity and yaw rate;
- joint position/velocity and previous action;
- stilt adapter pose/twist, foot/stilt tip contact flags, and local terrain/elevation samples;
- optional motion-reference features for an imitation warm-start.

**Actions**

- 37-DOF G1 joint-position targets for the native 37-DOF asset in the warm-up;
- the attached-stilt phase must preserve the same action interface unless the selected G1 variant requires a documented mapping.

**Reward groups**

- commanded velocity tracking;
- torso uprightness and angular-rate regularization;
- foot/stilt-tip support and foothold overlap;
- step timing / swing-foot clearance;
- action-rate, torque, acceleration and joint-limit penalties;
- fall and unsafe-contact penalties;
- optional motion imitation / AMP term, reported separately from task reward.

**Termination and success**

- torso/base contact or fall;
- stilt tip leaving the support region for a configured time;
- excessive tilt or joint-limit violation;
- success requires a full horizon without unsafe termination, commanded-speed tracking within tolerance, and a minimum number of valid alternating support events.

## Controls and ablations

| ID | Variant | Purpose |
|---|---|---|
| B0 | Isaac Lab native G1 flat velocity | Verify runtime and native controller learning. |
| B1 | G1 + attached stilts, no imitation | Isolate stilt dynamics and reward design. |
| B2 | B1 + foothold curriculum | Test sparse-support learning. |
| B3 | B2 + motion imitation warm-start | Test whether reference motion improves sample efficiency. |
| B4 | B2/B3 + disturbances and randomized adapter parameters | Test robustness and sim-to-real readiness. |

Keep initialization, physics step, action decimation, command ranges, seeds, and training budget fixed when comparing B1-B4. Report task reward and imitation reward separately.

## Primary evaluation

- success rate over at least 100 fixed-seed evaluation episodes per seed;
- fall rate and median time-to-fall;
- valid support-event count and left/right alternation rate;
- command velocity error and yaw error;
- peak tilt, peak joint-limit margin, contact impulse statistics;
- parameter randomization robustness;
- multi-step rollout videos and saved raw metrics.
