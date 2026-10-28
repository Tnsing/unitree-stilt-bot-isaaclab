#!/usr/bin/env python3
"""Held-out physical policy audit for G1-29 DOF and upstream dance imitation.

Records falls before autoreset and never substitutes reference replay for policy inference.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--skill',choices=['walk','run','dance','stilt'],required=True);p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--num_envs',type=int,default=64);p.add_argument('--steps',type=int,default=1500);p.add_argument('--seed',type=int,default=20261028);p.add_argument('--video',action='store_true');p.add_argument('--pushes',action='store_true');AppLauncher.add_app_launcher_args(p);args=p.parse_args();args.enable_cameras=args.video;app=AppLauncher(args).app
import hashlib,json,subprocess,types
import numpy as np
import torch,gymnasium as gym
import stilt_bot
from rsl_rl.runners import OnPolicyRunner
from isaaclab_tasks.utils.parse_cfg import load_cfg_from_registry
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.utils.io import dump_yaml
from isaaclab.utils.math import quat_error_magnitude,quat_apply
from stilt_bot.tasks.unitree_reference.env_cfg import ROOT


def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    out=args.output.resolve()
    if (out/'evaluation.json').exists():raise FileExistsError(out)
    out.mkdir(parents=True,exist_ok=True)
    task=f'UnitreeRef-G1-29dof-{args.skill.title()}-v0'
    cfg=load_cfg_from_registry(task,'env_cfg_entry_point');agent=load_cfg_from_registry(task,'rsl_rl_cfg_entry_point')
    cfg.scene.num_envs=args.num_envs;cfg.seed=args.seed;cfg.sim.device=args.device
    cfg.observations.policy.enable_corruption=False
    cfg.events.push_robot=cfg.events.push_robot if args.pushes else None
    if hasattr(cfg,'curriculum') and cfg.curriculum:
        cfg.curriculum.lin_vel_cmd_levels=None
    steps=args.steps;full_clip_steps=None;speed={'walk':.6,'run':1.8,'dance':0.,'stilt':.3}[args.skill]
    if args.skill!='dance':
        c=cfg.commands.base_velocity;c.ranges.lin_vel_x=(speed,speed);c.ranges.lin_vel_y=(0.,0.);c.ranges.ang_vel_z=(0.,0.);c.rel_standing_envs=0.;c.heading_command=False;c.rel_heading_envs=0.;c.debug_vis=False
        cfg.events.reset_base.params['pose_range']={'x':(0.,0.),'y':(0.,0.),'yaw':(0.,0.)}
    else:
        cfg.commands.motion.pose_range={};cfg.commands.motion.velocity_range={};cfg.commands.motion.joint_position_range=(0.,0.);cfg.commands.motion.debug_vis=False
        with np.load(cfg.commands.motion.motion_file) as data:full_clip_steps=len(data['joint_pos'])-2
        steps=min(steps,full_clip_steps)
    cfg.episode_length_s=(steps+2)*cfg.sim.dt*cfg.decimation
    dump_yaml(str(out/'env.yaml'),cfg);dump_yaml(str(out/'agent.yaml'),agent)
    env=RslRlVecEnvWrapper(gym.make(task,cfg=cfg,render_mode='rgb_array' if args.video else None));base=env.unwrapped
    runner=OnPolicyRunner(env,agent.to_dict(),log_dir=None,device=base.device);runner.load(str(args.checkpoint.resolve()),load_optimizer=False)
    policy=runner.get_inference_policy(device=base.device);robot=base.scene['robot'];sensor=base.scene.sensors['contact_forces']
    expr='.*_stilt_link' if args.skill=='stilt' else '.*ankle_roll_link';ids,names=robot.find_bodies(expr);cids,_=sensor.find_bodies(expr)
    motion=None
    if args.skill=='dance':
        motion=base.command_manager.get_term('motion')
        def fixed_start(self,env_ids):self.time_steps[env_ids]=0
        motion._adaptive_sampling=types.MethodType(fixed_start,motion)
    obs,_=env.reset();alive=torch.ones(args.num_envs,dtype=torch.bool,device=base.device)
    data={k:[] for k in ['root_pos','root_velocity','gravity','feet_pos','foot_force','valid','terminated','action','joint_pos']}
    if motion:data.update({k:[] for k in ['motion_frame','joint_error','anchor_error','body_error','body_rot_error']})
    proc=None
    try:
        for i in range(steps):
            with torch.inference_mode():
                action=policy(obs);obs,rew,done,extras=env.step(action)
            term=base.reset_terminated.clone();valid=alive & ~done.bool()
            values={'root_pos':robot.data.root_pos_w-base.scene.env_origins,'root_velocity':robot.data.root_lin_vel_w,'gravity':robot.data.projected_gravity_b,'feet_pos':robot.data.body_pos_w[:,ids]-base.scene.env_origins[:,None],'foot_force':sensor.data.net_forces_w[:,cids],'valid':valid,'terminated':term & alive,'action':action,'joint_pos':robot.data.joint_pos}
            if motion:values.update(motion_frame=motion.time_steps.clone(),joint_error=robot.data.joint_pos-motion.joint_pos,anchor_error=(motion.anchor_pos_w-motion.robot_anchor_pos_w).norm(dim=-1),body_error=(motion.body_pos_relative_w-motion.robot_body_pos_w).norm(dim=-1),body_rot_error=quat_error_magnitude(motion.body_quat_relative_w,motion.robot_body_quat_w))
            for k,v in values.items():data[k].append(v.detach().cpu().numpy().copy())
            alive &= ~term
            if args.video:
                target=robot.data.root_pos_w[0].detach().cpu().numpy();base.sim.set_camera_view(target+np.array([2.4,2.4,1.0]),target+np.array([0,0,-.15]))
                frame=base.render()
                if proc is None:
                    h,w=frame.shape[:2];proc=subprocess.Popen(['ffmpeg','-y','-loglevel','error','-f','rawvideo','-pixel_format','rgb24','-video_size',f'{w}x{h}','-framerate','50','-i','-','-an','-c:v','libx264','-crf','20','-pix_fmt','yuv420p',str(out/'demo.mp4')],stdin=subprocess.PIPE)
                proc.stdin.write(np.ascontiguousarray(frame[:,:,:3],dtype=np.uint8).tobytes())
            if i%250==0:print('AUDIT',args.skill,i,steps,int(alive.sum()),flush=True)
        arrays={k:np.stack(v) for k,v in data.items()};np.savez_compressed(out/'trajectory.npz',**arrays)
        valid=arrays['valid'];settled=valid & (np.arange(steps)[:,None]>.5/base.step_dt)
        fall=arrays['terminated'].any(axis=0);v=arrays['root_velocity'];pos=arrays['root_pos'];forces=np.linalg.norm(arrays['foot_force'],axis=-1);contact=forces>5
        rmse=float(np.sqrt(((v[:,:,0][settled]-speed)**2).mean())) if settled.any() else None
        result={'task':task,'checkpoint':str(args.checkpoint.resolve()),'checkpoint_sha256':sha(args.checkpoint.resolve()),'skill':args.skill,'num_envs':args.num_envs,'steps':steps,'dt':base.step_dt,'seed':args.seed,'survival_rate':float((~fall).mean()),'fall_rate':float(fall.mean()),'mean_speed_mps':float(v[:,:,0][settled].mean()) if settled.any() else None,'forward_rmse_mps':rmse,'p95_tilt_deg':float(np.percentile(np.rad2deg(np.arccos((-arrays['gravity'][:,:,2][valid]).clip(-1,1))),95)) if valid.any() else None,'support_body_names':names,'two_feet_unloaded_fraction':float((~contact.any(-1))[settled].mean()) if settled.any() else None,'upstream_commit':'2522f89ab8ad6b1188b0368d122d16cb0e3cb8c8','runtime':'Isaac Lab 2.3 / Isaac Sim 5.1 / PhysX compatibility port','stilt_geometry':args.skill=='stilt','pushes':args.pushes,'reference_replay':False,'boundary':'One held-out seed, nominal fixed command/full dance clip; not hardware or broad robustness evidence.'}
        # Flight requires both feet truly unloaded and a subsequent landing.
        flights=np.zeros(args.num_envs,dtype=int)
        for j in range(args.num_envs):
            start=None
            for i in range(steps):
                if not valid[i,j]:start=None;continue
                air=not contact[i,j].any()
                if air and start is None:start=i
                elif not air and start is not None:
                    if (i-start)*base.step_dt>=.06 and arrays['feet_pos'][start:i,j,:,2].min(axis=1).max()>.06:flights[j]+=1
                    start=None
        result['flight_event_count']=flights.tolist()
        if args.skill=='dance':
            joint_rmse=float(np.sqrt((arrays['joint_error'][settled]**2).mean())) if settled.any() else None
            body_error=float(arrays['body_error'][settled].mean()) if settled.any() else None
            result.update(joint_rmse_rad=joint_rmse,body_position_mean_error_m=body_error,anchor_mean_error_m=float(arrays['anchor_error'][settled].mean()) if settled.any() else None,reference_start_frame=0)
            passed=bool((~fall).mean()>=.95 and joint_rmse is not None and joint_rmse<.3 and body_error<.15)
            definition='Full-clip survival >=95%, per-joint RMSE <=0.3 rad, body position mean error <=0.15 m'
        else:
            passed=bool((~fall).mean()>=.95 and rmse is not None and rmse<=(.4 if args.skill=='run' else .25))
            definition='30 s survival >=95%, speed RMSE <=0.25 m/s (run <=0.4)'
            if args.skill=='run':passed=passed and (flights>=3).mean()>=.8;definition+='; >=3 flight+landing events in >=80% environments'
            if args.skill=='stilt':
                gains=pos[-1,:,0]-pos[0,:,0];load=arrays['foot_force'][:,:,:,2].sum(-1)
                result.update(stilt_forward_displacement_m=gains.tolist(),mean_stilt_vertical_support_N=float(load[settled].mean()) if settled.any() else None,stilt_spec=json.loads((ROOT/'assets/unitree_ros/g1_description/stilt_spec.json').read_text()))
                passed=passed and bool((gains>.5).mean()>=.8 and contact[settled].any())
                definition+='; >=0.5 m forward displacement in >=80% environments, actual stilt ground contact'
        horizon_complete=steps>=(full_clip_steps if args.skill=='dance' else 1500)
        result['horizon_complete']=horizon_complete
        result['gate']={'definition':definition,'passed':bool(passed and horizon_complete)}
        if proc:
            proc.stdin.close();assert proc.wait(timeout=60)==0;proc=None;result['video']=str(out/'demo.mp4');result['video_sha256']=sha(out/'demo.mp4')
        (out/'evaluation.json').write_text(json.dumps(result,indent=2,allow_nan=False));print(json.dumps(result,indent=2),flush=True)
    finally:
        if proc:proc.stdin.close();proc.wait(timeout=30)
        env.close()
try:main()
finally:app.close()
