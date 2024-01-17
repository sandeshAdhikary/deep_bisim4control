import src.envs.dmc2gym as dmc2gym
from src.utils import utils
from src.envs.gridworld import make_gridworld
import gym

def make_env(args):
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if args.domain_name == 'gridworld':
        size = 20
        env = make_gridworld(
            domain_name=args.domain_name,
            task_name=args.task_name,
            from_pixels=args.encoder_type.startswith('pixel'),
            seed=args.seed,
            height=args.image_size,
            width=args.image_size,
            size=size,
            episode_length=args.episode_length
        )
    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files if hasattr(args, 'resource_files') else None,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=args.encoder_type.startswith('pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat,
            episode_length=args.episode_length,
        )
        env.seed(args.seed)
        env.action_space.seed(args.seed)


    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)

    # the dmc2gym wrapper standardizes actions
    if isinstance(env.action_space, gym.spaces.box.Box):
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
    
    return env