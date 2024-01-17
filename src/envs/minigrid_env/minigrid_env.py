from minigrid.envs import FourRoomsEnv
from minigrid.wrappers import ImgObsWrapper
from gym import core, spaces
import numpy as np

def make_minigrid(*args, **kwargs):
    return MiniGridWrapper(*args, **kwargs)

    # env_config = {
    #     'agent_pos': (0,0),
    #     'goal_pos': (3,3),
    #     'max_steps': episode_length
    # }

    # if task_name == 'four_rooms':
    #     env = FourRoomsEnv(**env_config)
    # else:
    #     raise ValueError(f'Unknown task name: {task_name}')
    
    # if from_pixels:
    #     env = ImgObsWrapper(env)

    return env

class MiniGridWrapper(core.Env):

    def __init__(
        self,
        domain_name,
        task_name,
        resource_files,
        img_source,
        total_frames,
        from_pixels=False,
        height=84,
        width=84,
        frame_skip=1,
        environment_kwargs=None,
        seed=None,
        episode_length=100
    ):

        self._from_pixels = from_pixels
        self._height = height
        self._width = width
        self._frame_skip = frame_skip
        self._img_source = img_source


        environment_kwargs = environment_kwargs or {}
        env_config = {
            'agent_pos': environment_kwargs.get('agent_pos', (0,0)),
            'goal_pos': environment_kwargs.get('goal_pos', (3,3)),
            'max_steps': environment_kwargs.get('max_steps', 100)
        }
        if task_name == 'four_rooms':
            self._env = FourRoomsEnv(**env_config)
        else:
            raise ValueError(f'Unknown task name: {task_name}')


        # Use a normalized continuous action space
        self._true_action_space = self._env.action_space
        self._movement_actions = {
            0: 'turn_left',
            1: 'turn_right',
            2: 'move_forward'
        }
        
        # The action taken is the one with the highest value
        self._norm_action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(len(self._movement_actions),), # Only using movement actions
            dtype=np.float32
        )

        # create observation space
        if from_pixels:
            self._observation_space = spaces.Box(
                low=0, high=255, shape=[3, height, width], dtype=np.uint8
            )
        else:
            self._observation_space = self._env.observation_space


        self._internal_state_space = self._env.observation_space


if __name__ == "__main__":
    env = make_minigrid(
        domain_name='FourRooms',
        task_name='v0'
    )