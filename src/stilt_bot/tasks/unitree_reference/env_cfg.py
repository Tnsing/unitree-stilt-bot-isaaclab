"""G1-29 DOF reference tasks, explicit PhysX compatibility with Lab 2.3.

Upstream commit 2522f89; runtime backend is not advertised as Lab 3 / Newton.
"""
import importlib
import os
from pathlib import Path
from isaaclab.utils import configclass
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab import sim as sim_utils
from unitree_rl_lab.assets.robots.unitree import UnitreeUrdfFileCfg
from unitree_rl_lab.tasks.locomotion.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg as LocomotionRunner
from unitree_rl_lab.tasks.mimic.agents.rsl_rl_ppo_cfg import BasePPORunnerCfg as MimicRunner
from unitree_rl_lab.tasks.locomotion import mdp
from stilt_bot.tasks import phase_mdp

ROOT=Path(os.environ.get('UNITREE_REFERENCE_ROOT',Path(__file__).resolve().parents[4]))
ROBOT_DIR=ROOT/'assets/unitree_ros/g1_description'
VelocityCfg=importlib.import_module('unitree_rl_lab.tasks.locomotion.robots.g1.29dof.velocity_env_cfg').RobotFlatEnvCfg
DanceCfg=importlib.import_module('unitree_rl_lab.tasks.mimic.robots.g1_29dof.dance_102.tracking_env_cfg').RobotEnvCfg


def robot_spawn(stilts=False):
    return UnitreeUrdfFileCfg(asset_path=str(ROBOT_DIR/('g1_29dof_stilt_030.urdf' if stilts else 'g1_29dof_rev_1_0.urdf')),
        usd_dir=str(ROOT/'assets/generated'/('g1_stilt_030' if stilts else 'g1_29dof')),
        usd_file_name='robot.usd',merge_fixed_joints=not stilts,make_instanceable=True,
        replace_cylinders_with_capsules=False,
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0.,damping=0.)))


def visuals(cfg):
    cfg.viewer.origin_type='asset_root';cfg.viewer.asset_name='robot';cfg.viewer.env_index=0
    cfg.viewer.eye=(2.2,2.2,1.6);cfg.viewer.lookat=(0.,0.,.65);cfg.viewer.resolution=(1280,720)
    cfg.scene.terrain.visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.27,.30,.34))
    if hasattr(cfg.scene,'sky_light'):cfg.scene.sky_light.spawn.texture_file=None


@configclass
class ReferenceWalkCfg(VelocityCfg):
    def __post_init__(self):
        super().__post_init__();self.scene.robot.spawn=robot_spawn();visuals(self)
        self.commands.base_velocity.debug_vis=False
        self.commands.base_velocity.ranges.lin_vel_x=(.2,1.)
        self.commands.base_velocity.ranges.lin_vel_y=(-.15,.15)
        self.commands.base_velocity.ranges.ang_vel_z=(-.3,.3)
        self.commands.base_velocity.limit_ranges=self.commands.base_velocity.ranges
        self.curriculum.lin_vel_cmd_levels=None


@configclass
class ReferenceRunCfg(ReferenceWalkCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x=(1.5,2.2)
        self.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
        self.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
        self.rewards.gait.params.update(period=.60,threshold=.38)
        self.rewards.gait.weight=1.
        self.rewards.track_lin_vel_xy.weight=2.
        self.rewards.base_linear_velocity.weight=-.2
        self.rewards.action_rate.weight=-.01
        self.rewards.run_flight=RewTerm(func=phase_mdp.run_flight_reward,weight=.5,params={'period':.60,'sensor_cfg':SceneEntityCfg('contact_forces',body_names='.*ankle_roll.*'),'asset_cfg':SceneEntityCfg('robot',body_names='.*ankle_roll.*')})
        self.observations.policy.gait_phase=ObsTerm(func=mdp.gait_phase,params={'period':.60})
        self.observations.critic.gait_phase=ObsTerm(func=mdp.gait_phase,params={'period':.60})


@configclass
class ReferenceStiltCfg(ReferenceWalkCfg):
    def __post_init__(self):
        super().__post_init__();self.scene.robot.spawn=robot_spawn(stilts=True)
        self.scene.robot.init_state.pos=(0.,0.,1.12)
        self.commands.base_velocity.ranges.lin_vel_x=(.1,.5)
        self.commands.base_velocity.ranges.lin_vel_y=(-.05,.05)
        self.commands.base_velocity.ranges.ang_vel_z=(-.15,.15)
        self.rewards.base_height.params['target_height']=1.10
        self.terminations.base_height.params['minimum_height']=.75
        self.rewards.gait.params['period']=1.0
        for name in ('gait','feet_slide','feet_clearance'):
            term=getattr(self.rewards,name)
            for entity in term.params.values():
                if isinstance(entity,SceneEntityCfg) and entity.body_names:entity.body_names='.*_stilt_link'
        self.rewards.feet_clearance.params['target_height']=.435 # ankle-frame stilt link: .355 stance height plus .08 clearance
        self.rewards.undesired_contacts.params['sensor_cfg'].body_names=['(?!.*stilt_link).*']
        self.events.push_robot.params['velocity_range']={'x':(-.15,.15),'y':(-.15,.15)}
        self.scene.robot.spawn.articulation_props.solver_position_iteration_count=12
        self.viewer.eye=(2.5,2.5,1.8)


@configclass
class ReferenceDanceCfg(DanceCfg):
    def __post_init__(self):
        super().__post_init__();self.scene.robot.spawn=robot_spawn();visuals(self)
        self.commands.motion.motion_file=str(ROOT/'assets/motions/dance_102_wxyz_50hz.npz')
        self.commands.motion.debug_vis=False
        self.scene.contact_forces.debug_vis=False


@configclass
class ReferenceRunner(LocomotionRunner):
    def __post_init__(self):
        super().__post_init__();self.save_interval=100
        self.policy.actor_obs_normalization=False;self.policy.critic_obs_normalization=False
        self.obs_groups={'policy':['policy'],'critic':['critic']}
        self.experiment_name='unitree_reference_walk'

@configclass
class ReferenceRunRunner(ReferenceRunner):
    def __post_init__(self):
        super().__post_init__();self.experiment_name='unitree_reference_run'

@configclass
class ReferenceStiltRunner(ReferenceRunner):
    def __post_init__(self):
        super().__post_init__();self.experiment_name='unitree_reference_stilt'

@configclass
class ReferenceDanceRunner(MimicRunner):
    def __post_init__(self):
        super().__post_init__();self.experiment_name='unitree_reference_dance';self.save_interval=100
        self.policy.actor_obs_normalization=False;self.policy.critic_obs_normalization=False
        self.obs_groups={'policy':['policy'],'critic':['critic']}
