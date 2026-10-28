#!/usr/bin/env python3
"""Check imported stilt rigid bodies, masses, inertia, collision and actual contacts."""
import argparse
from isaaclab.app import AppLauncher
p=argparse.ArgumentParser();p.add_argument('--steps',type=int,default=300);AppLauncher.add_app_launcher_args(p);args=p.parse_args();app=AppLauncher(args).app
import json,numpy as np,torch
from pxr import Usd,UsdPhysics
from isaaclab import sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.scene import InteractiveScene,InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply
from stilt_bot.tasks.unitree_reference.env_cfg import ReferenceStiltCfg,ROOT
cfg=ReferenceStiltCfg()
@configclass
class SceneCfg(InteractiveSceneCfg):
    robot=cfg.scene.robot
    ground=AssetBaseCfg(prim_path='/World/ground',spawn=sim_utils.GroundPlaneCfg(physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.,dynamic_friction=1.)))
    contact_forces=ContactSensorCfg(prim_path='{ENV_REGEX_NS}/Robot/.*',history_length=3,track_air_time=True)
try:
    sim=sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=.005,device=args.device))
    scene=InteractiveScene(SceneCfg(num_envs=8,env_spacing=2.5));sim.reset()
    robot=scene['robot'];sensor=scene.sensors['contact_forces'];ids,names=robot.find_bodies('.*_stilt_link');cids,_=sensor.find_bodies('.*_stilt_link')
    assert robot.num_joints==29,(robot.num_joints,robot.joint_names)
    assert len(ids)==2,names
    masses=robot.root_physx_view.get_masses()[0];inertias=robot.root_physx_view.get_inertias()[0]
    mass_ok=bool(torch.allclose(masses[ids],torch.full_like(masses[ids],.8),atol=1e-4))
    expected=json.loads((ROOT/'assets/unitree_ros/g1_description/stilt_spec.json').read_text())
    inertia_matrices=inertias[ids].reshape(2,3,3)
    inertia_positive=bool((torch.linalg.eigvalsh(inertia_matrices)>0).all())
    inertia_matches=bool(torch.allclose(inertia_matrices.diagonal(dim1=-2,dim2=-1),torch.tensor(expected['diagonal_inertia_kg_m2']).expand(2,3),atol=1e-6))
    histories={k:[] for k in ['z','tip_z','stilt_force','ankle_force','tilt']}
    aids,_=sensor.find_bodies('.*ankle_roll_link')
    offset=torch.tensor([.035,0.,-.355],device=robot.device).expand(8,2,3)
    for i in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos);scene.write_data_to_sim();sim.step();scene.update(.005)
        tip=robot.data.body_pos_w[:,ids]+quat_apply(robot.data.body_quat_w[:,ids],offset)
        histories['z'].append(robot.data.root_pos_w[:,2].cpu().numpy().copy())
        histories['tip_z'].append(tip[:,:,2].cpu().numpy().copy())
        histories['stilt_force'].append(sensor.data.net_forces_w[:,cids].cpu().numpy().copy())
        histories['ankle_force'].append(sensor.data.net_forces_w[:,aids].cpu().numpy().copy())
        histories['tilt'].append(torch.acos((-robot.data.projected_gravity_b[:,2]).clamp(-1,1)).cpu().numpy().copy())
    data={k:np.stack(v) for k,v in histories.items()}; force=np.linalg.norm(data['stilt_force'],axis=-1)
    stage=sim.stage;collisions=[];fixed=[]
    for prim in Usd.PrimRange(stage.GetPseudoRoot(), Usd.TraverseInstanceProxies()):
        path=str(prim.GetPath())
        if '/env_0/' not in path:continue
        if prim.HasAPI(UsdPhysics.CollisionAPI) and 'stilt' in path:collisions.append(path)
        if prim.IsA(UsdPhysics.FixedJoint) and 'stilt' in path:fixed.append(path)
    report={'asset_spec':str(ROOT/'assets/unitree_ros/g1_description/stilt_spec.json'),'num_joints':robot.num_joints,'num_bodies':robot.num_bodies,'stilt_body_names':names,'stilt_masses_kg':masses[ids].tolist(),'stilt_inertias':inertias[ids].tolist(),'inertia_positive_definite':inertia_positive,'inertia_matches_analytic_spec':inertia_matches,'total_mass_kg':float(masses.sum()),'collision_prims':collisions,'fixed_joints':fixed,'max_stilt_force_N':float(force.max()),'mean_final_stilt_vertical_force_N':float(data['stilt_force'][-100:,:,:,2].sum(-1).mean()),'final_root_z_mean_m':float(data['z'][-1].mean()),'final_tilt_max_rad':float(data['tilt'][-1].max()),'minimum_tip_bottom_z_m':float(data['tip_z'].min()),'original_ankle_max_contact_force_N':float(np.linalg.norm(data['ankle_force'],axis=-1).max()),'default_pd_stable':bool(data['tilt'][-1].max()<.4),'passed':bool(mass_ok and inertia_positive and inertia_matches and len(collisions)>=4 and len(fixed)==2 and force.max()>10 and data['tip_z'].min()>-.04),'scope':'USD import + default-PD contact test; not trained stilt walking'}
    out=ROOT/'artifacts/unitree_reference/stilt_physics';out.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(out/'contact_trajectory.npz',**data);(out/'validation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
    if not report['passed']:raise RuntimeError('Stilt physics validation failed')
finally:
    # No live environment owns the stand-alone context; stop PhysX before closing Kit.
    if 'sim' in globals():
        sim.stop()
        sim.clear_all_callbacks()
        sim.clear_instance()
    app.close()
