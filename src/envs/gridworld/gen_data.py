from src.envs.gridworld import make_gridworld
from src.utils import utils
from itertools import permutations
import numpy as np
import torch
from PIL import Image
import imageio
from einops import rearrange

config = {
    'seed': 123,
    'domain_name': 'gridworld',
    'task_name': 'vert',
    'img_source': None,
    'episode_length': 1000,
    'image_size': 88,
    'frame_stack': 1,
    'num_envs': 1,
    'render': False,
    'total_frames': 1000,
    'encoder_type': 'pixel',
    'action_repeat': 1,
    'size': 20
}

env = make_gridworld(
    domain_name=config['domain_name'],
    task_name=config['task_name'],
    from_pixels=config['encoder_type'].startswith('pixel'),
    seed=config['seed'],
    height=config['image_size'],
    width=config['image_size'],
    size=config['size'],
    episode_length=config['episode_length'],
    random_init=False # Since we want to loop over all cells
)

env = utils.FrameStack(env, k=config['frame_stack'])


env.reset()
all_positions = permutations(range(env.size), 2)
data = []
for pos in all_positions:
    env.env.env.env.env.set_pos(np.array(pos))
    obs, rew, terminated, truncated, info = env.step(np.array([0.,0.]))
    data.append(
        {'obs': obs,
         'state': np.array(env.pos)}
    )
torch.save(data, 'gridworld_data.pt')

# Save the video of trajectory for inspection
imgs = [d['obs'] for d in data]



# Get the shape of the first image to set the video dimensions
# height, width, _ = image_list[0].shape

# Create a VideoWriter object
video_writer = imageio.get_writer('gridworld_video.mp4', fps=1)


# Convert and save each NumPy array image to the video
for img_array in imgs:
    img = img_array[:3,:]
    img = rearrange(img, 'c h w -> h w c')
    img = Image.fromarray(img)
    video_writer.append_data(np.array(img))

# Save the video
video_writer.close()
