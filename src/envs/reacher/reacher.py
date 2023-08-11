import mo_gymnasium as mogym
import gym
from typing import TypeVar
from os import path

import numpy as np
from gymnasium import utils
from gymnasium.envs.mujoco import MujocoEnv
from gymnasium.envs.mujoco.reacher_v4 import ReacherEnv
from gymnasium.spaces import Box, Discrete
from copy import copy
from src.utils.mo_gym_utils import LinearRewardWeightedVec
from src.utils.gym_utils import TransposeObservation

DEFAULT_CAMERA_CONFIG = {"trackbodyid": 0}


ObsType = TypeVar("ObsType")
ActType = TypeVar("ActType")

gym.envs.register(
    id='MORandomReacherEnv-v0',
    entry_point='src.envs.reacher.reacher:MORandomReacherEnv',
    # max_episode_steps=1_000,
)


def create_reacher_env(config):
    """
    Create an instance of the MO-reacher environment with the given config
    """
    # base_env = mogym.make('mo-reacher-v4', render_mode='rgb_array')
    model_path = path.join(ROOT_DIR, 'src/envs/reacher/mo_reacher.xml')
    base_env = mogym.make('MORandomReacherEnv-v0', 
                          max_episode_steps=config['max_steps'],
                          render_mode='rgb_array',
                          model_path=model_path,
                          frame_skip=config['frame_skip'],
                          img_size=config['img_size'],
                          )
    if config.get('img_observations'):
        # Get Image observations
        base_env = MOReacherImgObsWrapper(base_env)
        # Resize the image observations
        base_env = gym.wrappers.ResizeObservation(base_env, config['img_size'])
        # Transpose image observations from (h,w,c) -> (c,h,w)
        base_env = TransposeObservation(base_env)
    # Linearize the reward
    reward_weights = np.array(config['reward_weights']).astype(np.float32)
    if reward_weights.sum() > 0:
        reward_weights /= reward_weights.sum() # Make sure weights are normalized
    base_env = LinearRewardWeightedVec(base_env, weight=reward_weights)
    return base_env


class MORandomReacherEnv(ReacherEnv):
    """
    ## Description
    Mujoco version of `mo-reacher-v0`, based on [`Reacher-v4` environment](https://gymnasium.farama.org/environments/mujoco/reacher/).

    ## Observation Space
    The observation is 18-dimensional and contains:
    - sin and cos of the angles of the central and elbow joints
    - angular velocity of the central and elbow joints
    - (x,y,z) co-ordinates of the 4 targets

    ## Action Space
    The action space is discrete and contains the 3^2=9 possible actions based on applying positive (+1), negative (-1) or zero (0) torque to each of the two joints.

    ## Reward Space
    The reward is 4-dimensional and is defined based on the distance of the tip of the arm and the four target locations.
    For each i={1,2,3,4} it is computed as:
    ```math
        r_i = 1  - 4 * || finger_tip_coord - target_i ||^2
    ```
    """

    def __init__(self, **kwargs):
        utils.EzPickle.__init__(self, **kwargs)

        model_path = kwargs.pop('model_path')
        width = height = kwargs.pop('img_size')
        frame_skip = kwargs.pop('frame_skip')


        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(18,), dtype=np.float64)
        
        if model_path is None:
            model_path = self._default_model_path
        MujocoEnv.__init__(
            self,
            model_path,
            frame_skip,
            observation_space=self.observation_space,
            default_camera_config=DEFAULT_CAMERA_CONFIG,
            width=width,
            height=height,
            **kwargs,
        )
        actions = [-1.0, 0.0, 1.0]
        self.action_dict = dict()
        for a1 in actions:
            for a2 in actions:
                self.action_dict[len(self.action_dict)] = (a1, a2)
        self.action_space = Discrete(9)
        # Target goals: x1, y1, x2, y2, ... x4, y4
        self.orig_goal = np.array([0.14, 0.0, -0.14, 0.0, 0.0, 0.14, 0.0, -0.14])
        self.goal = copy(self.orig_goal)
        self.reward_space = Box(low=-np.inf, high=np.inf, shape=(4,))
        self.reward_dim = 4

    @property
    def _default_model_path(self):
        return path.join(mogym.__path__[0], 'envs/mujoco/assets/mo_reacher.xml')


    def step(self, a):
        real_action = self.action_dict[int(a)]
        vec_reward = np.array(
            [
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target1")[:2]),
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target2")[:2]),
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target3")[:2]),
                1 - 4 * np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target4")[:2]),
            ],
            dtype=np.float32,
        )

        self._step_mujoco_simulation(real_action, self.frame_skip)
        if self.render_mode == "human":
            self.render()

        ob = self._get_obs()


        target_dists = np.array(
            [
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target1")[:2]),
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target2")[:2]),
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target3")[:2]),
                np.linalg.norm(self.get_body_com("fingertip")[:2] - self.get_body_com("target4")[:2]),
            ],
            dtype=np.float32,
        )
        info = {
            'fingertip_pos' : self.get_body_com("fingertip")[:2].astype(np.float32),
            'target_dists': target_dists
            }

        return (
            ob,
            vec_reward,
            False,
            False,
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


class MOReacherImgObsWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env) -> None:
        if env.render_mode != "rgb_array":
            assert f"The render_mode must be rbg_array but it is {env.render_mode}"
        self.num_channels = 3
        super().__init__(env)
        self.observation_space = gym.spaces.Box(high=255, low=0, shape=(480, 480, 3), dtype='uint8')
    
    def observation(self, observation: ObsType) -> ObsType:
        return self.env.render()

