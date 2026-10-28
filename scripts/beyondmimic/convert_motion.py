#!/usr/bin/env python3
"""Convert upstream 60 Hz CSV into 50 Hz physical FK references for Lab 2.3.

This only prepares reference data. Teleporting the reference skeleton here is
explicitly not a policy rollout or a successful robot demonstration.
"""
import argparse
from pathlib import Path
from isaaclab.app import AppLauncher
parser=argparse.ArgumentParser();AppLauncher.add_app_launcher_args(parser);args=parser.parse_args()
app=AppLauncher(args).app
import hashlib,json,numpy as np,torch
from isaaclab import sim as sim_utils
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_slerp,quat_mul,quat_conjugate,axis_angle_from_quat
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from stilt_bot.tasks.beyondmimic.env_cfg import ROOT

@configclass
class SceneCfg(InteractiveSceneCfg):
    robot=G1_CYLINDER_CFG.replace(prim_path='{ENV_REGEX_NS}/Robot')
    robot.spawn.usd_dir=str(ROOT/'assets/generated/beyondmimic_g1')
    robot.spawn.usd_file_name='robot.usd'

try:
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.02,device=args.device))
    scene=InteractiveScene(SceneCfg(num_envs=1,env_spacing=2.));sim.reset()
    robot=scene['robot'];joint_ids,_=robot.find_joints(json.loads((ROOT/'assets/motions/dance_102_wxyz_50hz.json').read_text())['sdk_joint_order'],preserve_order=True)
    csv=ROOT/'third_party/unitree_rl_lab/source/unitree_rl_lab/unitree_rl_lab/tasks/mimic/robots/g1_29dof/dance_102/G1_Take_102.bvh_60hz.csv'
    raw=torch.tensor(np.loadtxt(csv,delimiter=','),dtype=torch.float32,device=robot.device)
    assert raw.shape[1]==36,raw.shape
    t=torch.arange(0,(len(raw)-1)/60.,.02,device=robot.device);f=t*60
    a=f.floor().long();b=(a+1).clamp(max=len(raw)-1);u=(f-a).unsqueeze(1)
    pose=raw[:,:3];q=raw[:,3:7][:,[3,0,1,2]] # CSV XYZW -> Lab 2.3 WXYZ
    assert torch.allclose(q.norm(dim=-1),torch.ones(len(q),device=q.device),atol=1e-3)
    pos=pose[a]*(1-u)+pose[b]*u
    quat=torch.stack([quat_slerp(q[i],q[j],blend.squeeze()) for i,j,blend in zip(a,b,u)])
    joints=raw[:,7:];jpos=joints[a]*(1-u)+joints[b]*u
    lin=torch.gradient(pos,spacing=.02,dim=0)[0];jvel=torch.gradient(jpos,spacing=.02,dim=0)[0]
    omega=axis_angle_from_quat(quat_mul(quat[2:],quat_conjugate(quat[:-2])))/.04
    omega=torch.cat([omega[:1],omega,omega[-1:]])
    history={k:[] for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
    for i in range(len(t)):
        root=torch.cat([pos[i],quat[i],lin[i],omega[i]])[None]
        jp=robot.data.default_joint_pos.clone();jv=torch.zeros_like(jp)
        jp[:,joint_ids]=jpos[i];jv[:,joint_ids]=jvel[i]
        robot.write_root_state_to_sim(root);robot.write_joint_state_to_sim(jp,jv)
        sim.forward();scene.update(.02)
        for k in history:history[k].append(getattr(robot.data,k)[0].detach().cpu().numpy().copy())
        if i%100==0:print('FK',i,len(t),flush=True)
    arrays={k:np.stack(v) for k,v in history.items()}
    for k,v in arrays.items():assert np.isfinite(v).all(),k
    # Verify articulated FK changed with the motion, rather than repeated cached transforms.
    assert np.std(arrays['body_pos_w'][:,robot.body_names.index('left_wrist_yaw_link')],axis=0).max()>.01
    out=ROOT/'assets/motions/beyondmimic_dance_102_50hz.npz';out.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out,fps=np.array(50),**arrays)
    manifest={'reference_only':True,'source_csv':str(csv),'source_sha256':hashlib.sha256(csv.read_bytes()).hexdigest(),'output':str(out),'output_sha256':hashlib.sha256(out.read_bytes()).hexdigest(),'frames':len(t),'fps':50,'quaternion_order':'wxyz','body_names':robot.body_names,'joint_names':robot.joint_names,'sdk_joint_order':json.loads((ROOT/'assets/motions/dance_102_wxyz_50hz.json').read_text())['sdk_joint_order'],'shapes':{k:list(v.shape) for k,v in arrays.items()}}
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2));print(json.dumps(manifest,indent=2))
finally:
    # No live environment owns the stand-alone context; stop PhysX before closing Kit.
    if 'sim' in globals():
        sim.stop()
        sim.clear_all_callbacks()
        sim.clear_instance()
    app.close()
