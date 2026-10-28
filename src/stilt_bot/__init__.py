"""Project-owned Isaac Lab task registrations.

Legacy StiltBot tasks keep their native G1 assets for old checkpoint compatibility.
UnitreeRef tasks use the pinned Unitree 29-DOF description; the Stilt task includes
physical attached stilts. Task versions keep incompatible checkpoints separate.
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

# Unitree isaaclab3.0 reference extensions. Runtime/asset provenance is kept separately.
for _skill,_env,_runner in [
    ('Walk','ReferenceWalkCfg','ReferenceRunner'),
    ('Run','ReferenceRunCfg','ReferenceRunRunner'),
    ('Dance','ReferenceDanceCfg','ReferenceDanceRunner'),
    ('Stilt','ReferenceStiltCfg','ReferenceStiltRunner'),
]:
    _register(f'UnitreeRef-G1-29dof-{_skill}-v0',
        f'stilt_bot.tasks.unitree_reference.env_cfg:{_env}',
        f'stilt_bot.tasks.unitree_reference.env_cfg:{_runner}')
