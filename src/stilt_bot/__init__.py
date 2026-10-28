"""Project-owned Isaac Lab task registrations.

All registered tasks use Unitree's native G1 humanoid asset on flat ground.  No
stilt geometry is registered in this phase.  The four formal skills are kept as
separate Gym IDs so their logs, checkpoints and videos cannot be confused.
"""

import gymnasium as gym


def _register(task_id: str, env_cfg: str, runner_cfg: str) -> None:
    gym.register(
        id=task_id,
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": env_cfg,
            "rsl_rl_cfg_entry_point": runner_cfg,
        },
    )


_register(
    "StiltBot-G1-Warmup-v0",
    "stilt_bot.tasks.stilt_g1_env_cfg:StiltG1WarmupEnvCfg",
    "stilt_bot.tasks.stilt_g1_env_cfg:G1FlatPPORunnerCfg",
)
_register(
    "StiltBot-G1-Walk-v0",
    "stilt_bot.tasks.skill_g1_env_cfg:G1WalkEnvCfg",
    "stilt_bot.tasks.skill_g1_env_cfg:G1WalkPPORunnerCfg",
)
_register(
    "StiltBot-G1-Run-v0",
    "stilt_bot.tasks.skill_g1_env_cfg:G1RunEnvCfg",
    "stilt_bot.tasks.skill_g1_env_cfg:G1RunPPORunnerCfg",
)
_register(
    "StiltBot-G1-Jump-v0",
    "stilt_bot.tasks.skill_g1_env_cfg:G1JumpEnvCfg",
    "stilt_bot.tasks.skill_g1_env_cfg:G1JumpPPORunnerCfg",
)
_register(
    "StiltBot-G1-Dance-v0",
    "stilt_bot.tasks.skill_g1_env_cfg:G1DanceEnvCfg",
    "stilt_bot.tasks.skill_g1_env_cfg:G1DancePPORunnerCfg",
)

# Versioned registrations preserve V0 checkpoint compatibility and failed controls.
for _skill in ['Run','Jump','Dance']:
    _register(
        f'StiltBot-G1-{_skill}-v1',
        f'stilt_bot.tasks.phase_g1_env_cfg:G1{_skill}V1EnvCfg',
        f'stilt_bot.tasks.phase_g1_env_cfg:G1{_skill}V1PPORunnerCfg',
    )
