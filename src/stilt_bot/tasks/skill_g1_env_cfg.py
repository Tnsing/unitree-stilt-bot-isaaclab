"""Unitree G1 walk, run, jump and dance configurations.

No stilt geometry is added here.  The robot is Isaac Lab's native Unitree G1
minimal asset.  Walk/run are native command-tracking tasks; jump/dance are
phase-conditioned skill tasks with explicit synthetic references, not claims of
motion-capture imitation.
"""
from __future__ import annotations

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.flat_env_cfg import G1FlatEnvCfg
from isaaclab_tasks.manager_based.locomotion.velocity.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg

from . import skill_mdp


@configclass
class G1SkillPPORunnerCfg(G1FlatPPORunnerCfg):
    """Shared PPO settings for the four formal skill runs."""


@configclass
class G1WalkPPORunnerCfg(G1SkillPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "stiltbot_g1_walk_formal"


@configclass
class G1RunPPORunnerCfg(G1SkillPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "stiltbot_g1_run_formal"


@configclass
class G1JumpPPORunnerCfg(G1SkillPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "stiltbot_g1_jump_formal"


@configclass
class G1DancePPORunnerCfg(G1SkillPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "stiltbot_g1_dance_formal"


@configclass
class G1WalkEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.2, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.25, 0.25)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.6, 0.6)
        self.commands.base_velocity.ranges.heading = (-1.57, 1.57)


@configclass
class G1RunEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.commands.base_velocity.ranges.lin_vel_x = (1.0, 2.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.2, 0.2)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)
        self.commands.base_velocity.ranges.heading = (-1.0, 1.0)
        self.rewards.feet_air_time.weight = 1.0
        self.rewards.feet_air_time.params["threshold"] = 0.25
        self.rewards.action_rate_l2.weight = -0.003


@configclass
class G1JumpEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 12.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.jump_height_phase = RewTerm(
            func=skill_mdp.root_height_phase_tracking,
            weight=5.0,
            params={
                "period": 1.6,
                "base_height": 0.74,
                "amplitude": 0.22,
                "height_std": 0.08,
                "velocity_std": 0.65,
            },
        )
        self.rewards.jump_airborne = RewTerm(
            func=skill_mdp.airborne_phase_reward,
            weight=1.0,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link"),
                "period": 1.6,
                "flight_start": 0.30,
                "flight_end": 0.72,
            },
        )
        self.rewards.flat_orientation_l2.weight = -2.0
        self.rewards.lin_vel_z_l2.weight = -0.02
        self.rewards.ang_vel_xy_l2.weight = -0.04
        self.rewards.action_rate_l2.weight = -0.004


@configclass
class G1DanceEnvCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 20.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        self.rewards.track_lin_vel_xy_exp = None
        self.rewards.track_ang_vel_z_exp = None
        self.rewards.feet_air_time = None
        self.rewards.dance_body_phase = RewTerm(
            func=skill_mdp.dance_root_tracking,
            weight=3.0,
            params={
                "period": 2.4,
                "lateral_velocity": 0.45,
                "yaw_velocity": 0.8,
                "base_height": 0.74,
                "velocity_std": 0.45,
                "height_std": 0.10,
            },
        )
        self.rewards.dance_joint_wave = RewTerm(
            func=skill_mdp.joint_wave_tracking,
            weight=1.5,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_pitch_joint",
                    ],
                ),
                "period": 2.4,
                "amplitude": 0.35,
                "position_std": 0.30,
                "phase_offset": 0.0,
            },
        )
        self.rewards.flat_orientation_l2.weight = -1.5
        self.rewards.lin_vel_z_l2.weight = -0.2
        self.rewards.ang_vel_xy_l2.weight = -0.04
        self.rewards.action_rate_l2.weight = -0.004
