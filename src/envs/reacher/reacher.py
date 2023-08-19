import mo_gymnasium as mogym
import gym
from typing import TypeVar
from os import path

import numpy as np
from gym.envs.mujoco.reacher_v4 import ReacherEnv
from gym.envs.mujoco import MujocoEnv
from gymnasium import utils
from gym.spaces import Box, Discrete
from copy import copy
from PIL import Image
from einops import rearrange

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

class MOReacher(ReacherEnv):
    """
    ## Description
    Mujoco version of `mo-reacher-v0`, based on [`Reacher-v4` environment](https://gymnasium.farama.org/environments/mujoco/reacher/).

    ## Observation Space
    The observation is 18-dimensional and contains:
    - sin and cos of the angles of the central and elbow joints
    - angular velocity of the central and elbow joints
    - (x,y,z) co-ordinates of the 4 targets

    ## Action Space
    # 2-dimensional continuos action specifying torque [-1,1] on each of the two joints

    ## Reward Space
    The reward is 4-dimensional and is defined based on the distance of the tip of the arm and the four target locations.
    For each i={1,2,3,4} it is computed as:
    ```math
        r_i = 1  - 4 * || finger_tip_coord - target_i ||^2
    ```
    """

    def __init__(self, config, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        model_path = config.get('model_path')
        width = height = config.get('img_size', 256)
        frame_skip = config.get('frame_skip', 2)
        self.max_episode_steps = config.get('max_episode_steps', 100)
        self.goal_weights = config.get('goal_weights', [-1.0, 1.0, 0.0, 0.0])
        

        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        
        if model_path is None:
            model_path = self._default_model_path
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=self.observation_space,
            # default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=width,
            height=height,
            **kwargs,
        )
        self.action_space = Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        # Target goals: x1, y1, x2, y2, ... x4, y4
        self.orig_goal = np.array([0.14, 0.0, -0.14, 0.0, 0.0, 0.14, 0.0, -0.14])
        self.goal = copy(self.orig_goal)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(4,))
        self.reward_dim = 4

        self.render_mode = 'rgb_array'
        

    @property
    def _default_model_path(self):
        return path.join(mogym.__path__[0], 'envs/mujoco/assets/mo_reacher.xml')


    def step(self, a):
        self.num_steps += 1
        real_action = np.clip(a, self.action_space.low, self.action_space.high)

        target_dists = np.array(
            [
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target1")[:2]),
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target2")[:2]),
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target3")[:2]),
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target4")[:2]),
            ],
            dtype=np.float32,
        )
        vec_reward = 1- 4*target_dists

        self._step_mujoco_simulation(real_action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()

        info = {
            'fingertip_pos' : self.get_body_com("fingertip")[:2].astype(np.float32),
            'target_dists': target_dists
            }
        
        scalar_reward = np.dot(self.goal_weights, vec_reward)

        truncated = self.num_steps >= self.max_episode_steps

        terminated = False

        return (
            ob,
            scalar_reward,
            terminated,
            truncated,
            info,
        )

    def reset_model(self):

        # Change the goal positions
        # self.goal = self.np_random.uniform(low=-0.1, high=0.1, size=len(self.orig_goal)) + self.orig_goal
        self.goal = self.np_random.uniform(low=-0.2, high=0.2, size=len(self.orig_goal)) + self.orig_goal

        qpos = self.np_random.uniform(low=-0.1, high=0.1, size=self.model.nq) + self.init_qpos
        qpos[:2] = np.array([0, 3.1415 / 2])  # init position
        qpos[-len(self.goal) :] = self.goal
        qvel = self.init_qvel + self.np_random.uniform(low=-0.005, high=0.005, size=self.model.nv)
        qvel[-2:] = 0
        self.set_state(qpos, qvel)
        self.num_steps = 0
        return self._get_obs()

    def _get_obs(self):
        theta = self.data.qpos.flat[:2]
        agent_info = np.concatenate(
            [
                np.cos(theta),
                np.sin(theta),
                self.data.qvel.flat[:2] * 0.1,
            ]
        )
        target_info = np.concatenate([self.get_body_com("fingertip") - self.get_body_com(f"target{d}") for d in range(1,5)])
        ob = np.concatenate([agent_info, target_info])
        return ob


class MOReacherRGB(gym.ObservationWrapper):
    def __init__(self, config) -> None:
        super().__init__(env=MOReacher(config))
        self.img_size = config.get('img_size', 84)
        self.img_mode = config.get('img_mode', 'CHW')

        if self.img_mode == 'CHW':
            obs_shape = (3, self.img_size, self.img_size)
        elif self.img_mode == 'HWC':
            obs_shape = (self.img_size, self.img_size, 3)
        else:
            raise ValueError(f"Invalid image mode {self.img_mode}")

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)
    
    
    def observation(self, observation):
        return self.get_obs()

    def get_obs(self):
        # Return the RGB array of the grid
        obs = self.render()
        obs = self.resize_obs(obs)
        if self.img_mode == 'CHW':
            obs = rearrange(obs, 'h w c -> c h w')
        return obs

    def resize_obs(self, obs):
        """
        Resize the observations if needed
        """
        if obs.shape[0] == obs.shape[1] == self.img_size:
            return obs
        return np.array(Image.fromarray(obs).resize((self.img_size, self.img_size)))