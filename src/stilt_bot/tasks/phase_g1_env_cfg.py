"""V1 fixes: phase is observable, run rewards permit flight, jump penalizes drift."""
from isaaclab.managers import ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.utils import configclass
from .skill_g1_env_cfg import G1RunEnvCfg,G1JumpEnvCfg,G1DanceEnvCfg,G1SkillPPORunnerCfg
from . import phase_mdp


def sensor():return SceneEntityCfg('contact_forces',body_names='.*_ankle_roll_link')
def feet():return SceneEntityCfg('robot',body_names='.*_ankle_roll_link')


def make_observable(cfg,period):
    cfg.observations.policy.skill_clock=ObsTerm(func=phase_mdp.clock,params={'period':period})
    cfg.observations.policy.root_local=ObsTerm(func=phase_mdp.root_local)
    cfg.observations.policy.feet_contact=ObsTerm(func=phase_mdp.contact_observation,params={'sensor_cfg':sensor()})
    cfg.commands.base_velocity.debug_vis=False


@configclass
class G1RunV1EnvCfg(G1RunEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        make_observable(self,.60)
        self.rewards.feet_air_time=None # Native positive-biped reward explicitly favors single support.
        self.rewards.run_contact_phase=RewTerm(func=phase_mdp.run_contact_reward,weight=2.,params={'period':.60,'sensor_cfg':sensor()})
        self.rewards.run_flight_phase=RewTerm(func=phase_mdp.run_flight_reward,weight=1.,params={'period':.60,'sensor_cfg':sensor(),'asset_cfg':feet()})
        self.rewards.track_lin_vel_xy_exp.weight=2.
        self.commands.base_velocity.ranges.lin_vel_x=(1.5,2.2)
        self.commands.base_velocity.ranges.lin_vel_y=(0.,0.)
        self.commands.base_velocity.ranges.ang_vel_z=(0.,0.)
        self.commands.base_velocity.heading_command=False
        self.commands.base_velocity.ranges.heading=None
        self.commands.base_velocity.rel_standing_envs=0.
        self.commands.base_velocity.rel_heading_envs=0.
        self.events.reset_base.params['pose_range']={'x':(0.,0.),'y':(0.,0.),'yaw':(0.,0.)}


@configclass
class G1JumpV1EnvCfg(G1JumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        make_observable(self,1.6)
        self.observations.policy.jump_reference=ObsTerm(func=phase_mdp.jump_reference,params={'period':1.6})
        self.rewards.jump_height_phase=RewTerm(func=phase_mdp.jump_tracking,weight=6.,params={'period':1.6})
        self.rewards.jump_airborne=RewTerm(func=phase_mdp.jump_contact_reward,weight=2.,params={'period':1.6,'sensor_cfg':sensor()})
        self.rewards.horizontal_stability=RewTerm(func=phase_mdp.horizontal_stability,weight=-1.5)
        self.rewards.local_drift=RewTerm(func=phase_mdp.local_drift,weight=-3.)
        self.rewards.joint_deviation_arms.weight=-.3
        self.rewards.joint_deviation_fingers.weight=-.15
        self.rewards.dof_torques_l2.weight=-5.e-7
        self.commands.base_velocity.ranges.heading=None
        self.commands.base_velocity.rel_heading_envs=0.
        self.commands.base_velocity.rel_standing_envs=0.
        self.events.reset_base.params['pose_range']={'x':(0.,0.),'y':(0.,0.),'yaw':(0.,0.)}


@configclass
class G1DanceV1EnvCfg(G1DanceEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        make_observable(self,2.4)
        self.observations.policy.dance_targets=ObsTerm(func=phase_mdp.dance_targets,params={'period':2.4})
        self.rewards.joint_deviation_arms=None # Do not oppose the moving arm reference.
        self.rewards.dance_joint_wave.weight=3.
        self.rewards.dance_body_phase.weight=5.
        self.rewards.local_drift=RewTerm(func=phase_mdp.local_drift,weight=-.2)
        self.commands.base_velocity.ranges.heading=None
        self.commands.base_velocity.rel_heading_envs=0.
        self.commands.base_velocity.rel_standing_envs=0.
        self.events.reset_base.params['pose_range']={'x':(0.,0.),'y':(0.,0.),'yaw':(0.,0.)}


@configclass
class G1RunV1PPORunnerCfg(G1SkillPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__();self.experiment_name='stiltbot_g1_run_v1';self.save_interval=100

@configclass
class G1JumpV1PPORunnerCfg(G1SkillPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__();self.experiment_name='stiltbot_g1_jump_v1';self.save_interval=100

@configclass
class G1DanceV1PPORunnerCfg(G1SkillPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__();self.experiment_name='stiltbot_g1_dance_v1';self.save_interval=100
