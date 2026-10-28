#!/usr/bin/env python3
"""Independent physical rollout audit. Run with Isaac Sim Python, never substitutes reward for success."""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument('--skill', choices=['walk','run','jump','dance'], required=True)
parser.add_argument('--task', default=None)
parser.add_argument('--checkpoint', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--num_envs', type=int, default=64)
parser.add_argument('--steps', type=int, default=1000)
parser.add_argument('--seed', type=int, default=20261027)
parser.add_argument('--video', action='store_true')
from isaaclab.app import AppLauncher
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = args.video
app = AppLauncher(args).app

import hashlib
import json
import math
import os
import subprocess
import numpy as np
import torch
import gymnasium as gym
import stilt_bot
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from skill_audit_metrics import summarize


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    if (out/'evaluation.json').exists():
        raise FileExistsError(f'Refusing to overwrite completed evidence: {out}')
    task = args.task or f'StiltBot-G1-{args.skill.title()}-v0'
    cfg = load_cfg_from_registry(task,'env_cfg_entry_point')
    agent = load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs = args.num_envs
    cfg.seed = args.seed
    cfg.sim.device = args.device or 'cuda:0'
    cfg.observations.policy.enable_corruption = False
    # Fixed held-out forward command, synchronized phase and nominal reset. No randomized episode lengths.
    command = cfg.commands.base_velocity
    command.heading_command = False
    command.rel_standing_envs = 0.0
    command.rel_heading_envs = 0.0
    command.debug_vis = False
    speed = {'walk':0.6,'run':1.8,'jump':0.0,'dance':0.0}[args.skill]
    command.ranges.lin_vel_x = (speed,speed)
    command.ranges.lin_vel_y = (0.0,0.0)
    command.ranges.ang_vel_z = (0.0,0.0)
    command.ranges.heading = None
    cfg.events.reset_base.params['pose_range'] = {'x':(0.0,0.0),'y':(0.0,0.0),'yaw':(0.0,0.0)}
    cfg.viewer.origin_type = 'asset_root'
    cfg.viewer.asset_name = 'robot'
    cfg.viewer.env_index = 0
    cfg.viewer.eye = (2.8,2.8,1.6)
    cfg.viewer.lookat = (0.0,0.0,0.0)
    cfg.viewer.resolution = (960,720)
    # One complete 20 s audit; mark auto-reset samples invalid and never call them survival.
    cfg.episode_length_s = max(cfg.episode_length_s,args.steps*cfg.sim.dt*cfg.decimation)
    dump_yaml(str(out/'eval_env.yaml'), cfg)
    dump_yaml(str(out/'eval_agent.yaml'), agent)
    env = RslRlVecEnvWrapper(gym.make(task,cfg=cfg,render_mode='rgb_array' if args.video else None))
    base = env.unwrapped
    runner = OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=base.device)
    runner.load(str(args.checkpoint.resolve()), load_optimizer=False)
    policy = runner.get_inference_policy(device=base.device)
    robot = base.scene['robot']
    feet_ids, feet_names = robot.find_bodies('.*_ankle_roll_link')
    sensor = base.scene.sensors['contact_forces']
    contact_ids,contact_names = sensor.find_bodies('.*_ankle_roll_link')
    arm_ids,arm_names = robot.find_joints('.*_shoulder_pitch_joint|.*_shoulder_roll_joint|.*_shoulder_yaw_joint|.*_elbow_pitch_joint')
    obs = env.get_observations()
    alive = torch.ones(args.num_envs,device=base.device,dtype=torch.bool)
    history = {k:[] for k in ['position','velocity','angular_velocity','gravity','feet_position','force','arm_position','valid','terminated','time_out','phase_time']}
    proc = None
    try:
        for step in range(args.steps):
            with torch.inference_mode():
                obs, reward, done, extras = env.step(policy(obs))
            terminated = base.reset_terminated.clone()
            time_out = base.reset_time_outs.clone()
            valid = alive & ~done.bool()
            if step == args.steps-1:
                # Timeout is not a physical fall but its reset-state samples are excluded.
                pass
            fields = {
                'position':robot.data.root_pos_w-base.scene.env_origins,
                'velocity':robot.data.root_lin_vel_w,
                'angular_velocity':robot.data.root_ang_vel_w,
                'gravity':robot.data.projected_gravity_b,
                'feet_position':robot.data.body_pos_w[:,feet_ids,:]-base.scene.env_origins[:,None,:],
                'force':sensor.data.net_forces_w[:,contact_ids,:],
                'arm_position':robot.data.joint_pos[:,arm_ids]-robot.data.default_joint_pos[:,arm_ids],
                'valid':valid,'terminated':terminated & alive,'time_out':time_out & alive,
                'phase_time':base.episode_length_buf*base.step_dt,
            }
            for k,v in fields.items():
                history[k].append(v.detach().cpu().numpy().copy())
            alive &= ~terminated
            if args.video:
                frame = base.render()
                if proc is None:
                    h,w = frame.shape[:2]
                    proc = subprocess.Popen(['ffmpeg','-loglevel','error','-n','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate',str(round(1/base.step_dt)),'-i','-','-an','-c:v','libx264','-pix_fmt','yuv420p',str(out/'demo.mp4')],stdin=subprocess.PIPE)
                proc.stdin.write(np.ascontiguousarray(frame[:,:,:3],dtype=np.uint8).tobytes())
            if step%200==0:
                print(f'AUDIT {args.skill} step={step}/{args.steps} alive={alive.sum().item()}/{args.num_envs}',flush=True)
        data = {k:np.stack(v) for k,v in history.items()}
        np.savez_compressed(out/'trajectory.npz',**data)
        result = summarize(data,args.skill,base.step_dt,speed)
        result.update({'task':task,'seed':args.seed,'num_envs':base.num_envs,'steps':args.steps,'dt':base.step_dt,
            'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':digest(args.checkpoint.resolve()),
            'feet_body_names':feet_names,'contact_body_names':contact_names,'arm_joint_names':arm_names,
            'observation_terms':base.observation_manager.active_terms,'reset_exclusion':'physical terminal and auto-reset samples excluded',
            'stilt_geometry':False,'asset_usd':cfg.scene.robot.spawn.usd_path,
            'reference':'synthetic phase targets' if args.skill in ['jump','dance'] else 'fixed held-out velocity command',
            'video':str(out/'demo.mp4') if args.video else None})
        if proc:
            proc.stdin.close(); rc=proc.wait(timeout=60);proc=None
            if rc: raise RuntimeError(f'ffmpeg exit {rc}')
            result['video_sha256']=digest(out/'demo.mp4')
        (out/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
        print(json.dumps(result,indent=2),flush=True)
    finally:
        if proc:
            proc.stdin.close();proc.wait(timeout=60)
        env.close()

try:
    main()
finally:
    app.close()
