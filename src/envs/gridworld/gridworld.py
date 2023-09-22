# import gymnasium as gym
import gym
import numpy as np
from itertools import product
import itertools
from PIL import Image, ImageDraw
from copy import copy
from matplotlib import cm
from copy import copy
from PIL import Image
from einops import rearrange

class GridWorld(gym.Env):

    def __init__(self, config):
        super().__init__()

        size = config['size']
        goals= config['goals']
        obstacles=config['obstacles']
        init_pos=config.get('init_pos', None)
        goal_weights=config.get('goal_weights', [1.0]*len(goals))
        obstacle_weights=config.get('obstacle_weights', [1.0]*len(goals))
        reward_mode=config.get('reward_mode', 'sparse')
        seed=config.get('seed', 321)
        self.rgb_image_bkg = None
        self.action_mode = config.get('action_mode', 'discrete')
        self.img_mode = config.get('img_mode', 'CHW')
        self.random_init = config.get('random_init', False)
        self.sticky_radius = config.get('sticky_radius', 0.0)
        
        # Fixed params
        self.cell_size = 10
        self.goal_scale = 5.
        self.obstacle_scale = 5.
        
        assert self.img_mode in ['CHW', 'HWC']

        self.height = self.width = self.size = size
        if self.action_mode == 'discrete':
            # Action space: [0,1,2,3] = [right, left, up, down]
            self.action_space = gym.spaces.Discrete(4)
        elif self.action_mode == 'continuous':
            # Action space: 2D action clipped within [-1,1]. Defines the amount of movement in each direction
            self.action_space = gym.spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        else:
            raise ValueError(f"Invalid action mode {self.action_mode}")
        
        # Observation: agent position
        self.observation_space = gym.spaces.Box(low=np.array([0,0]), high=np.array([self.width, self.height]), shape=(2,), dtype=int)
        self.default_init_pos = init_pos if init_pos else np.array([0, 0])
        self.reward_mode = reward_mode

        self.goals = np.array(goals).reshape(-1,2) if goals is not None else None
        self.goal_weights = np.array(goal_weights) if goal_weights is not None else None
        if self.goals is not None and self.goal_weights is None:
            self.goal_weights = np.ones(self.goals.shape[0])/self.goals.shape[0]
        self.obstacles = np.array(obstacles).reshape(-1,2) if obstacles is not None else None
        self.obstacle_weights = np.array(obstacle_weights) if obstacle_weights is not None else None
        if self.obstacles is not None and self.obstacle_weights is None:
            self.obstacle_weights = np.ones(self.obstacles.shape[0])/self.obstacles.shape[0]
        
        self._sanity_checks()

    def _sanity_checks(self):
        if self.goals is not None:
            assert self.goals.shape[1] == 2
            assert np.all(self.goals >= 0)
            assert np.all(self.goals < self.size)
        

        if self.obstacles is not None:
            assert self.obstacles.shape[1] == 2
            assert np.all(self.obstacles >= 0)
            assert np.all(self.obstacles < self.size)
        

    def reset(self, seed=None, options=None):
        
        options = options if options is not None else {}
        init_pos = options.get('init_pos', None) # If init_pos provided, use it

        if (init_pos is None) and self.random_init:
            init_pos = np.random.randint(0, self.size, size=2) # If random_init, select random init_pos
        
        self.pos = init_pos if (init_pos is not None) else copy(self.default_init_pos) # Else, use default init_pos
        self.pos = self.pos.astype(np.float32)

        if not isinstance(self.pos, np.ndarray):
            self.pos = np.array(self.pos)
        
        assert all(0 <= self.pos) and all(self.pos < self.size)
        self.steps = 0
        return self.get_obs(), {}

    def set_pos(self, pos):
        self.pos = pos

    def set_default_init_pos(self, pos):
        self.default_init_pos = pos

    @property
    def action_aliases(self):
        return {0:'right', 1: 'left', 2: 'up', 3: 'down'}


    def step(self, action):
        # If discrete: Action space = [0,1,2,3] = [right, left, up, down]
        # If continuous: Action space: [0,1] = right, (1,2] = left, (2,3] = up, (3,4] = down, anything else: no-op

        self.steps += 1
        if self.action_mode == 'discrete':
            if action == 0:
                # right
                self.pos[1] += 1
            elif action == 1:
                # left
                self.pos[1] -= 1
            elif action == 2:
                # up
                self.pos[0] -= 1
            elif action == 3:
                # down
                self.pos[0] += 1
            else:
                raise ValueError("Invalid action")
        elif self.action_mode == 'continuous':

            action = np.round(np.clip(action, -1, 1))

            self.pos[0] += action[0]
            self.pos[1] += action[1]
        else:
            raise ValueError(f"Invalid action mode {self.action_mode}")

        self.pos = np.clip(self.pos, [1,1], [self.width-1, self.height-1])

        obs = self.get_obs()
        rew, rew_info = self._get_reward(self.pos)
        truncated = False # Truncation, if needed, should be handled by TimeLimit wrapper
        terminated = np.any(np.linalg.norm(self.pos - self.goals, axis=1) < 1e-1) # if close to any goal, terminate
        info = rew_info

        return obs, rew, terminated, truncated, info
    

    def get_obs(self):
        return self.pos

    def _get_reward(self, pos, goal_bandwidth=500.0, obstacle_bandwidth=500.0):
        """
        Positive reward based on sum of distances to goals
        Negative reward based on sum of distances to obstacles
        """
        rew_info = {}
        if self.reward_mode == 'dense':
            rew = 0.0
            rew_info['goal_reward'] = 0.0
            if self.goals is not None:
                # Get goal reward
                goal_dists = ((pos.reshape(1,2) - self.goals)**2).sum(axis=1)
                goal_dists = goal_dists/(2*self.size**2)
                rews = np.exp(-goal_bandwidth*goal_dists)
                goal_rew = np.dot(rews,self.goal_weights/self.goal_weights.sum()) # sum over goals
                goal_rew = goal_rew * self.goal_scale
                rew_info['goal_reward'] = goal_rew
                rew += goal_rew
            
            rew_info['obstacle_reward'] = 0.0
            if self.obstacles is not None:
                obstacle_dists = ((pos.reshape(1,2) - self.obstacles)**2).sum(axis=1)
                obstacle_dists = obstacle_dists/(2*self.size**2)
                rews = np.exp(-obstacle_bandwidth*obstacle_dists)
                obstacle_rew = -np.dot(rews,self.obstacle_weights/self.obstacle_weights.sum()) # sum over goals
                obstacle_rew = obstacle_rew * self.obstacle_scale
                rew_info['obstacle_reward'] = obstacle_rew
                rew += obstacle_rew

        elif self.reward_mode == 'sparse':
            rew = 0.0
            rew_info['goal_reward'] = 0.0
            if self.goals is not None:
                at_goal = np.all(pos.reshape(1,2) == self.goals, axis=1)
                goal_rew = (at_goal * self.goal_weights).sum() * self.goal_scale
                rew_info['goal_reward'] = goal_rew
                rew += goal_rew
            rew_info['obstacle_reward'] = 0.0
            if self.obstacles is not None:
                at_obstacle = np.all(pos.reshape(1,2) == self.obstacles, axis=1)
                obstacle_rew = -(at_obstacle * self.obstacle_weights).sum() * self.obstacle_scale
                rew_info['obstacle_reward'] = obstacle_rew
                rew += obstacle_rew
        else:
            raise ValueError("Invalid reward mode")
        

        # Add penalty for each step
        rew -= 1.
        rew_info['goal_reward'] -= 1.
        rew_info['obstacle_reward'] -= 1.

        # Create vec_reward
        rew_info['vec_reward'] = [rew_info[x] for x in ['goal_reward', 'obstacle_reward']]

        return rew, rew_info

    def render(self, **kwargs):
        if self.render_mode is None or self.render_mode == 'rgb_array':
            return self.render_rgbarray()
        else:
            raise NotImplementedError("Only rgb_array render mode is supported")

    def render_rgbarray(self, pos=None, return_pil_image=False):
        if self.rgb_image_bkg is None:
            self.pre_render_rgbarray()
        
        image = copy(self.rgb_image_bkg)
        draw = ImageDraw.Draw(image)
        
        pos = self.pos if pos is None else pos

        # Add agent
        x_start = self.cell_size
        y_start = self.cell_size
        center = (np.array([pos[1], pos[0]]) * self.cell_size) + [x_start + self.cell_size/2, y_start + self.cell_size/2]
        agent_size = 1.0*self.cell_size/8
        color = '#656dc6'

        top_left = (center[0] - agent_size, center[1] - agent_size)
        bottom_right = (center[0] + agent_size, center[1] + agent_size)

        draw.rectangle(
            [top_left, 
             bottom_right],
             fill=color,
             outline=(255, 255, 255),
             width=max(1, int(self.cell_size*0.1))
        )
        del draw
        if return_pil_image:
            return image
        return np.array(image)

    def pre_render_rgbarray(self):

        # Create a blank image
        image_size = self.cell_size*self.size + 2*self.cell_size
        image_size = image_size + (image_size % 16) # Make sure image size is a multiple of 16
        image = Image.new("RGB", (image_size, image_size), color=(0, 0, 0))

        # Draw some lines
        draw = ImageDraw.Draw(image)
        y_start = self.cell_size
        # y_end = image.height - self.cell_size

        # for x in range(self.cell_size, image.width, self.cell_size):
        #     line = ((x, y_start), (x, y_end))
        #     draw.line(line, fill='#b3adad')

        x_start = self.cell_size
        # x_end = image.width - self.cell_size

        # for y in range(self.cell_size, image.height, self.cell_size):
        #     line = ((x_start, y), (x_end, y))
        #     draw.line(line, fill='#b3adad')

        if self.goals is not None:
            for goal in self.goals:
                center = (np.array([goal[1], goal[0]]) * self.cell_size) + [x_start + self.cell_size/2, y_start + self.cell_size/2]
                radius = 1.0*self.cell_size/4
                color = '#347c2a'
                draw.ellipse([(center[0] - radius, center[1] - radius),
                            (center[0] + radius, center[1] + radius)],
                            fill=color, outline=color)
            
        if self.obstacles is not None:
            for obstacle in self.obstacles:
                center = (np.array([obstacle[1], obstacle[0]]) * self.cell_size) + [x_start + self.cell_size/2, y_start + self.cell_size/2]
                radius = 1.0*self.cell_size/4
                color = '#d76320'
                draw.ellipse([(center[0] - radius, center[1] - radius),
                            (center[0] + radius, center[1] + radius)],
                            fill=color, outline=color)

        del draw
        self.rgb_image_bkg = image
        return image

    def render_heatmap(self, heatmap, color_mode='discrete'):
        assert len(heatmap) == self.size

        image = Image.new("RGB", (self.cell_size*self.size + 2*self.cell_size, self.cell_size*self.size + 2*self.cell_size), color=(255, 255, 255))
        draw = ImageDraw.Draw(image)

        # Define a color mapping
        if color_mode == 'discrete':
            num_values = len(np.unique(heatmap))
            if num_values <= 10:
                cmap = 'tab10'
            elif num_values <= 20:
                cmap = 'tab20'
            else:
                raise NotImplementedError("Only 20 discrete colors are supported")
            colormap = cm.get_cmap(cmap, num_values)
        elif color_mode == 'continuous':
            colormap = cm.get_cmap('viridis')
        else:
            raise NotImplementedError("Only discrete and continuous color modes are supported")

        heatmap_normalizer = (np.max(heatmap) - np.min(heatmap))
        heatmap_normalizer = 1 if heatmap_normalizer <= 0 else heatmap_normalizer
        norm_heatmap = (heatmap - np.min(heatmap)) / heatmap_normalizer

        # # Iterate through the array and draw cells with the corresponding colors
        for i in range(self.size):
            for j in range(self.size):
                cell_value = norm_heatmap[i][j]
                color = tuple(int(255 * c) for c in colormap(cell_value)[:3])  # Scale colormap values to 0-255
                r, c = i+1, j+1
                draw.rectangle([(c * self.cell_size ,
                                 r * self.cell_size ), 
                                 ((c + 1) * self.cell_size, 
                                  (r + 1) * self.cell_size )], 
                                  fill=color)

        return image

    def viz_distance_fn(self, distance_fn):
        pass

    @property
    def state_iterator(self):
        return product(range(self.height), range(self.width))
    
    @property
    def action_iterator(self):
        return range(self.action_space.n)


class GridWorldRGB(gym.ObservationWrapper):
    
    def __init__(self, config):
        super().__init__(env=GridWorld(config))
        self.img_size = config.get('img_size', 28)
        if self.img_mode == 'CHW':
            obs_shape = (3, self.img_size, self.img_size)
        elif self.img_mode == 'HWC':
            obs_shape = (self.img_size, self.img_size, 3)
        else:
            raise ValueError(f"Invalid image mode {self.img_mode}")

        self.observation_space = gym.spaces.Box(low=0, high=255, shape=obs_shape, dtype=np.uint8)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        return obs, info

    def observation(self, observation):
        return self.get_obs()
        
    def get_obs(self):
        # Return the RGB array of the grid
        obs = self.render_rgbarray()
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
        return np.array(Image.fromarray(obs).resize((self.img_size, self.img_size), Image.ANTIALIAS))

    def _get_reward(self, pos):
        return super()._get_reward(pos)
    

class GridWorldRGB_Boxed(gym.Wrapper):
    """
    This is the same as GridWorldRGB, except when computing rewards.
    We split up the grid into k equal boxes. When computing the rewards,
    we compute use the centroid of the box that the agent is in, instead
    of the agent's actual position.
    
    e.g. say we have 20x20 grid, that is split into boxes of size 5x5.
    So we have 4 rows and 4 columns of boxes. If the agent is in cell
    (2,3), it is in Box number 1. So it's approx position is the centroid
    of the first box, which is (3,3)
    """

    def __init__(self, config):
        super().__init__(env=GridWorldRGB(config))
        self.env.env.sticky_radius = 5.0


