"""Reward terms for the non-stilt Unitree G1 skill tasks.

The walk/run tasks use Isaac Lab's native velocity-tracking rewards.  Jump and
 dance use deterministic phase-conditioned reference signals so that the
 training/evaluation protocol is reproducible without pretending that an
 external motion-capture dataset is available.
"""
from __future__ import annotations

import math

import torch

from isaaclab.managers import SceneEntityCfg


def _phase(env, period: float) -> torch.Tensor:
    time = env.episode_length_buf.to(torch.float32) * env.step_dt
    return 2.0 * math.pi * torch.remainder(time, period) / period


def root_height_phase_tracking(
    env,
    period: float,
    base_height: float,
    amplitude: float,
    height_std: float,
    velocity_std: float,
) -> torch.Tensor:
    """Track a smooth crouch-to-air-to-landing vertical cycle."""
    asset = env.scene["robot"]
    phase = _phase(env, period)
    target_height = base_height + 0.5 * amplitude * (1.0 - torch.cos(phase))
    target_velocity = 0.5 * amplitude * (2.0 * math.pi / period) * torch.sin(phase)
    height_error = asset.data.root_pos_w[:, 2] - target_height
    velocity_error = asset.data.root_lin_vel_w[:, 2] - target_velocity
    return torch.exp(-torch.square(height_error) / (height_std**2)) * torch.exp(
        -torch.square(velocity_error) / (velocity_std**2)
    )


def airborne_phase_reward(env, sensor_cfg: SceneEntityCfg, period: float, flight_start: float, flight_end: float):
    """Reward both feet being unloaded during the flight portion of a jump."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    phase = torch.remainder(env.episode_length_buf.to(torch.float32) * env.step_dt, period) / period
    in_flight = (phase >= flight_start) & (phase <= flight_end)
    net_force = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    feet_in_contact = (net_force > 1.0).any(dim=1)
    return (~feet_in_contact & in_flight).to(torch.float32)


def dance_root_tracking(
    env,
    period: float,
    lateral_velocity: float,
    yaw_velocity: float,
    base_height: float,
    velocity_std: float,
    height_std: float,
) -> torch.Tensor:
    """Track a repeatable lateral/yaw body rhythm while staying upright."""
    asset = env.scene["robot"]
    phase = _phase(env, period)
    target_vy = lateral_velocity * torch.sin(phase)
    target_wz = yaw_velocity * torch.sin(2.0 * phase)
    vy_error = asset.data.root_lin_vel_w[:, 1] - target_vy
    wz_error = asset.data.root_ang_vel_w[:, 2] - target_wz
    height_error = asset.data.root_pos_w[:, 2] - base_height
    return (
        torch.exp(-torch.square(vy_error) / (velocity_std**2))
        * torch.exp(-torch.square(wz_error) / (velocity_std**2))
        * torch.exp(-torch.square(height_error) / (height_std**2))
    )


def joint_wave_tracking(
    env,
    asset_cfg: SceneEntityCfg,
    period: float,
    amplitude: float,
    position_std: float,
    phase_offset: float = 0.0,
) -> torch.Tensor:
    """Track a bounded sinusoidal wave on selected joints around G1 defaults."""
    asset = env.scene[asset_cfg.name]
    phase = _phase(env, period) + phase_offset
    joint_ids = asset_cfg.joint_ids
    current = asset.data.joint_pos[:, joint_ids]
    default = asset.data.default_joint_pos[:, joint_ids]
    # Alternating signs produce a simple left/right arm and torso rhythm.
    signs = torch.ones(len(joint_ids), device=current.device)
    signs[1::2] = -1.0
    target = default + amplitude * torch.sin(phase).unsqueeze(1) * signs.unsqueeze(0)
    return torch.exp(-torch.mean(torch.square(current - target), dim=1) / (position_std**2))
