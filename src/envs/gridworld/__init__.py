import gym
from gym.envs.registration import register
import numpy as np
import itertools

def make_gridworld(
        domain_name,
        task_name, # ['circles','horz', 'vert', 'random]
        from_pixels=False,
        seed=123,
        frame_skip=1,
        episode_length=60,
        height=84,
        width=84,
        size=10,
        sparse_rewards=False,
        random_init=True,
):
    
    env_id = f'{domain_name}-{task_name}-v1'

    max_episode_steps = (episode_length + frame_skip - 1) // frame_skip

    # Define goals and obstacles based on task name
    goals, goal_weights, obstacles, obstacle_weights = get_goals_and_obstacles(size=size, mode=task_name)

    env_config = {
        'size': size,
        'init_pos': None,
        'goals': goals,
        'obstacles': obstacles,
        'goal_weights': goal_weights,
        'obstacle_weights': obstacle_weights,
        'reward_mode': 'sparse' if sparse_rewards else 'dense',
        'seed': seed,
        'img_size': height,
        'action_mode': 'continuous',
        'random_init': random_init
    }
    assert height == width, 'Height and width must be equal'
    

    if not env_id in gym.envs.registry:
        
        if from_pixels:
            entry_point = 'src.envs.gridworld.gridworld:GridWorldRGB'
        else:
            entry_point = 'src.envs.gridworld.gridworld:GridWorld'

        register(
            id=env_id,
            entry_point=entry_point,
            kwargs={'config': env_config},
        )
    
    return gym.make(env_id, max_episode_steps=max_episode_steps)

def get_goals_and_obstacles(size, mode='diag'):
    """
    Generates goals and obstacles based on the mode
    """

    if mode == 'diag':
        goal_mid = 15
        goal_sep = 3
        goals = np.array([
            [goal_mid-2*goal_sep, goal_mid-goal_sep], [goal_mid-goal_sep, goal_mid], [goal_mid,goal_mid+goal_sep],
            [goal_mid-goal_sep,goal_mid-2*goal_sep], [goal_mid, goal_mid-goal_sep], [goal_mid+goal_sep,goal_mid],
            ])
        goals = np.clip(goals, a_min=0, a_max=size-1)
        goal_weights = [1]*len(goals)

        obs_mid = 5
        obs_sep = 3
        obstacles = np.array([
            [obs_mid-2*obs_sep, obs_mid-obs_sep], [obs_mid-obs_sep, obs_mid], [obs_mid,obs_mid+obs_sep],
            [obs_mid-obs_sep,obs_mid-2*obs_sep], [obs_mid, obs_mid-obs_sep], [obs_mid+obs_sep,obs_mid],
            ])
        obstacles = np.clip(obstacles, a_min=0, a_max=size-1)
        obstacle_weights = [1]*len(obstacles) 

    elif mode == 'diag_two':
        goal_mid = 15
        goal_sep = 3
        goals = np.array([
            [goal_mid-2*goal_sep, goal_mid-goal_sep], [goal_mid-goal_sep, goal_mid], [goal_mid,goal_mid+goal_sep],
            [goal_mid-goal_sep,goal_mid-2*goal_sep], [goal_mid, goal_mid-goal_sep], [goal_mid+goal_sep,goal_mid],
            ])
        goals = np.clip(goals, a_min=0, a_max=size-1)
        goal_weights = [5, 5, 5, 1, 1, 1]

        obs_mid = 5
        obs_sep = 3
        obstacles = np.array([
            [obs_mid-2*obs_sep, obs_mid-obs_sep], [obs_mid-obs_sep, obs_mid], [obs_mid,obs_mid+obs_sep],
            [obs_mid-obs_sep,obs_mid-2*obs_sep], [obs_mid, obs_mid-obs_sep], [obs_mid+obs_sep,obs_mid],
            ])
        obstacles = np.clip(obstacles, a_min=0, a_max=size-1)

        obstacle_weights = [1, 1, 1, 5, 5, 5]
    elif mode == 'vert':
        mid = int(size/2)
        sep = int(size/4)
        num = 10

        goals = np.zeros((10,2))
        goals[:,1] = mid - sep
        goals[:,0] = range(sep, sep+num)
        goals = np.clip(goals, 0, size-2)
        goals = np.unique(goals, axis=0)

        obstacles = np.zeros((10,2))
        obstacles[:,1] = mid + sep
        obstacles[:,0] = range(sep, sep+num)
        obstacles = np.clip(obstacles, 0, size-2)
        obstacles = np.unique(obstacles, axis=0)

        goal_weights = [1]*len(goals)
        obstacle_weights = [1]*len(obstacles)
    elif mode == 'horz':
        mid = int(size/2)
        sep = int(size/4)
        num = 10
        goals = np.zeros((10,2))
        goals[:,0] = mid - sep
        goals[:,1] = range(sep, sep+num)
        goals = np.clip(goals, 0, size-2)
        goals = np.unique(goals, axis=0)

        obstacles = np.zeros((10,2))
        obstacles[:,0] = mid + sep
        obstacles[:,1] = range(sep, sep+num)
        obstacles = np.clip(obstacles, 0, size-2)
        obstacles = np.unique(obstacles, axis=0)

        goal_weights = [1]*len(goals)
        obstacle_weights = [1]*len(obstacles)
    elif mode == 'random':
        np.random.seed(321)
        num = 4
        all_points = np.array(list(itertools.product(range(size), range(size))))
        goals = None
        goals = all_points[np.random.choice(len(all_points), size=num, replace=False)]
        obstacles = all_points[np.random.choice(len(all_points), size=num, replace=False)]
        goals = np.clip(goals, 0, size-1)
        goals = np.unique(goals, axis=0)
        obstacles = np.clip(obstacles, 0, size-1)
        obstacles = np.unique(obstacles, axis=0)


        goal_weights = [1]*len(goals)
        obstacle_weights = [1]*len(obstacles)
        
        # goal_weights = [1.0, 1.0, 10.0, 10.0]
        # obstacle_weights = [1.0, 1.0, 10.0, 10.0]

    elif mode == 'circles':
        goals = get_cirlce_points([int(size/2),int(size/2)], 2, 20)
        obstacles = get_cirlce_points([int(size/2),int(size/2)], 6, 20)
        goals = np.clip(goals, 0, size-1)
        obstacles = np.clip(obstacles, 0, size-1)
        goals = np.unique(goals, axis=0)
        obstacles = np.unique(obstacles, axis=0)
        goal_weights = [1]*len(goals)
        obstacle_weights = [1]*len(obstacles)
    elif mode == 'debug':
        mid = int(size/2)
        goals = np.array([[mid, mid]]).reshape(-1,2)
        goal_weights = [1]*len(goals)
        goals = np.clip(goals, 0, size-1)
        goals = np.unique(goals, axis=0)
        obstacles = None
        obstacle_weights = None
    elif mode == 'simple':
        goals = np.array([2,3])
        obstacles = np.array([5, 5])
        goal_weights = [1.]
        obstacle_weights = [1.]
    else:
        raise ValueError(f"Mode {mode} not recognized")

    return goals, goal_weights, obstacles, obstacle_weights

def get_cirlce_points(center, radius, num_points=20):
    # Generate coordinates of the circle points
    circle_points = []
    for theta in np.linspace(0, 2*np.pi, num_points):
        x = center[0] + radius * np.cos(theta)
        y = center[1] + radius * np.sin(theta)
        circle_points.append((np.round(x), np.round(y)))
    return np.array(circle_points[:-1])