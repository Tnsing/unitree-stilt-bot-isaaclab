"""Official BeyondMimic tracking MDP, isolated Lab 2.3 compatibility configuration."""
from pathlib import Path
from isaaclab.utils import configclass
from isaaclab import sim as sim_utils
from whole_body_tracking.tasks.tracking.config.g1.flat_env_cfg import G1FlatEnvCfg
from whole_body_tracking.tasks.tracking.config.g1.agents.rsl_rl_ppo_cfg import G1FlatPPORunnerCfg
ROOT = Path(__file__).resolve().parents[4]

@configclass
class BeyondMimicDanceCfg(G1FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = str(ROOT/'assets/motions/beyondmimic_dance_102_50hz.npz')
        self.commands.motion.debug_vis = False
        self.scene.contact_forces.debug_vis = False
        self.scene.robot.spawn.usd_dir = str(ROOT/'assets/generated/beyondmimic_g1')
        self.scene.robot.spawn.usd_file_name = 'robot.usd'
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(.27,.30,.34))
        self.viewer.resolution = (1280,720)

@configclass
class BeyondMimicDanceRunner(G1FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = 'beyondmimic_dance'
        self.save_interval = 500
        # RSL-RL 3 replaces the 2.x runner empirical_normalization field.
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.obs_groups = {'policy':['policy'], 'critic':['critic']}
