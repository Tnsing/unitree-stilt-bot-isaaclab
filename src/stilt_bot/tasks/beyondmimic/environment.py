from isaaclab.envs import ManagerBasedRLEnv
from .reset_sync import refresh_motion_cache

class BeyondMimicTrackingEnv(ManagerBasedRLEnv):
    """Fix full-reset reference cache on Lab 2.3; normal step ordering unchanged."""
    def reset(self, **kwargs):
        _, extras = super().reset(**kwargs)
        refresh_motion_cache(self.command_manager.get_term('motion'))
        return self.observation_manager.compute(update_history=False), extras
