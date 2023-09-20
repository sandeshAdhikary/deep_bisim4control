from gym import Wrapper
import local_dm_control_suite as suite
from local_dm_control_suite.reacher import Reacher
from dmc2gym import make
from dm_control.utils import rewards
import numpy as np

class ReacherLevelSet(Wrapper):
    """
    Wrapper around the DMC reacher env. The observation image is split into
    evely cized cells to form a grid. The centers of cells are assigned to be
    cell-centroids. The reward at each step is computed based on the position of
    the closest cell-centroid instead of the actual position
    """

    def __init__(self, env, config=None):
        self.env = env

        # Set the centroid positions
        # TODO: self.tau should be bigger than goal tolerance
        self.levelset_factor = config.get('levelset_factor', 1.0)

    def step(self, action):
        obs, _, terminated, truncated, info = self.env.step(action)
        reward = self.get_reward(self.env.physics)
        return  obs, reward, terminated, truncated, info
    

    def get_reward(self, physics):        
        finger_pos = physics.named.data.geom_xpos['finger', :2]
        target_pos = physics.named.data.geom_xpos['target', :2]
        target_dist = np.linalg.norm(finger_pos - target_pos)

        # Replace target_dist with the mid-point of its level set
        goal_radius =  physics.named.model.geom_size['target', 0]
        tau = self.levelset_factor * goal_radius
        if target_dist > goal_radius:
            # use distance to mid point of level set
            target_dist = ((target_dist // tau) * tau) + tau/2.0

        radii = goal_radius +  physics.named.model.geom_size['finger', 0]
        return rewards.tolerance(target_dist, (0, radii), margin=0.1)

    
