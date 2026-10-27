"""Project-owned Isaac Lab task registrations.

The warm-up task intentionally reuses Isaac Lab's native Unitree G1 asset and PPO
configuration. The attached-stilt task is not registered until its USD and contact
validation are complete.
"""

import gymnasium as gym


gym.register(
    id="StiltBot-G1-Warmup-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": "stilt_bot.tasks.stilt_g1_env_cfg:StiltG1WarmupEnvCfg",
        "rsl_rl_cfg_entry_point": "stilt_bot.tasks.stilt_g1_env_cfg:G1FlatPPORunnerCfg",
    },
)
