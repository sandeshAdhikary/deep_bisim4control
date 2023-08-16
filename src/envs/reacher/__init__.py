import gym
from gym.envs.registration import register
import os

def make_reacher(
        domain_name,
        task_name, # []
        from_pixels=False,
        seed=123,
        frame_skip=1,
        episode_length=75,
        height=84,
        width=84,
        environment_kwargs=None
):
    env_id = f'{domain_name}-{task_name}-v1'

    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip
    robot_model_path = os.path.join(os.path.dirname(__file__), 'mo_reacher.xml')

    env_config = {
        'model_path': robot_model_path,
        'img_size': height,
        'max_episode_steps': max_episode_steps,
        # 'frame_skip': frame_skip
    }

    assert height == width, 'Height and width must be equal'

    if not env_id in gym.envs.registry:

        if from_pixels:
            entry_point = 'envs.reacher.reacher:MOReacherRGB'
        else:
            entry_point = 'envs.reacher.reacher:MOReacher'

        register(
            id=env_id,
            entry_point=entry_point,
            kwargs={'config': env_config},
        )
    
    return gym.make(env_id)