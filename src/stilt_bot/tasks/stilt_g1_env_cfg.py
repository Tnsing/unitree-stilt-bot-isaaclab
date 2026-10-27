"""Unitree G1 warm-up configuration for the stilt-bot project.

This is deliberately a native flat-ground warm-up. It is not the attached-stilt
physics task; keeping it explicit prevents a baseline from being mislabeled as
stilt walking.
"""

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg


@configclass
class StiltG1WarmupEnvCfg(G1FlatEnvCfg):
    """Stable G1 warm-up before adding the attached stilt adapter."""

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 10.0
        self.scene.num_envs = 64
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.6)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.15, 0.15)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)
        self.rewards.feet_air_time.weight = 0.9
        self.rewards.feet_slide.weight = -0.2
        self.rewards.action_rate_l2.weight = -0.008
        self.rewards.dof_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=[".*_hip_.*", ".*_knee_joint", ".*_ankle_.*"]
        )
