from copy import copy, deepcopy
import src.dmc2gym as dmc2gym
from src.envs.gridworld import make_gridworld
from src.envs.gridworld.callbacks import GridWorldEvalCallback
from src.envs.dmc2gym.callbacks import DMCCallback
from src.envs.distractor_wrappers import DistractorWrapper
from src.utils import utils
from src.envs.vec_wrappers import VecEnvWrapper
import gym
from functools import partial
import glob
from types import SimpleNamespace
VEC_ENV_TYPES = (VecEnvWrapper, gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv, gym.vector.VectorEnv)

def setup_env(args):
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    env = make_env(args)

    # if args.num_train_envs > 1:
    env = VecEnvWrapper([partial(make_env, args) for _ in range(args.num_train_envs)])

    
    if args.eval_img_sources is not None:
        assert len(args.eval_img_sources) == (args.num_eval_envs - 1), "Num eval envs must be len(eval_img_sources)+1"
        assert all([x in ['color', 'noise', 'mnist', 'driving_stereo'] for x in args.eval_img_sources]), "Unknown eval_img_sources"
        # The first eval_env uses same config as train env
        env_fns = [partial(make_env, args)]
        for idx in range(args.num_eval_envs-1):
            args_new = deepcopy(args)
            args_new.seed =  args.seed + idx # diff seed for eval
            args_new.img_source = args.eval_img_sources[idx]
            env_fns.append(partial(make_env, args_new))
        eval_env = VecEnvWrapper(env_fns)
    else:
        eval_env = VecEnvWrapper([partial(make_env, args) for _ in range(args.num_eval_envs)])

    assert isinstance(env, VEC_ENV_TYPES)
    domain_callback = make_domain_callback(args)
    return env, eval_env, domain_callback


def make_env_fn(args):
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if args.domain_name == 'carla':
        env = CarlaEnv(
            render_display=args.render,  # for local debugging only
            display_text=args.render,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=args.image_size,
            max_episode_steps=1000,
            frame_skip=args.action_repeat,
            is_other_cars=True,
            port=args.port
        )
        # TODO: implement env.seed(args.seed) ?
    elif args.domain_name == 'gridworld':
        size = 10
        env = make_gridworld(
            domain_name=args.domain_name,
            task_name=args.task_name,
            from_pixels=args.encoder_type.startswith('pixel'),
            seed=args.seed,
            height=args.image_size,
            width=args.image_size,
            size=size,
            boxed_env=args.boxed_env,
            episode_length=args.episode_length

        )
        # TODO: Initialize seed
    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=args.encoder_type.startswith('pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat,
            episode_length=args.episode_length
        )
        env.seed(args.seed)
        env.action_space.seed(args.seed)

    if args.distractor in ['ideal_gas']:
        distractor_kwargs = {
            'num_bodies': int(args.distraction_level*20),
            'num_dimensions': 2,
            'distractor_type': args.distractor_type,
            'img_shrink_factor': args.distractor_img_shrink_factor
        }
        env = DistractorWrapper(env, copy(distractor_kwargs))
        # TODO: Initialize seed for distractor

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)

    # the dmc2gym wrapper standardizes actions
    if isinstance(env.action_space, gym.spaces.box.Box):
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
    
    return env

def make_domain_callback(args):
    if args.domain_name == 'carla':
        domain_callback = None
    elif args.domain_name == 'gridworld':
        domain_callback = None
        # domain_callback = GridWorldEvalCallback()
    else:
        domain_callback = DMCCallback({
            'domain_name': args.domain_name,
            'task_name': args.task_name,
        })
    
    return domain_callback


def setup_env_trainer(args):
    
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if args.num_envs > 1:
        env_fns = [partial(make_env, args) for _ in range(args.num_envs)]
    else:
        env_fns = partial(make_env, args)
    
    return env_fns


def make_env(args):
    from types import SimpleNamespace
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if args.domain_name == 'carla':
        env = CarlaEnv(
            render_display=args.render,  # for local debugging only
            display_text=args.render,  # for local debugging only
            changing_weather_speed=0.1,  # [0, +inf)
            rl_image_size=args.image_size,
            max_episode_steps=1000,
            frame_skip=args.action_repeat,
            is_other_cars=True,
            port=args.port
        )
        # TODO: implement env.seed(args.seed) ?
    elif args.domain_name == 'gridworld':
        size = 10
        env = make_gridworld(
            domain_name=args.domain_name,
            task_name=args.task_name,
            from_pixels=args.encoder_type.startswith('pixel'),
            seed=args.seed,
            height=args.image_size,
            width=args.image_size,
            size=size,
            boxed_env=args.boxed_env,
            episode_length=args.episode_length

        )
        # TODO: Initialize seed
    else:
        env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=args.encoder_type.startswith('pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat,
            episode_length=args.episode_length
        )
        env.seed(args.seed)
        env.action_space.seed(args.seed)

    if args.distractor in ['ideal_gas']:
        distractor_kwargs = {
            'num_bodies': int(args.distraction_level*20),
            'num_dimensions': 2,
            'distractor_type': args.distractor_type,
            'img_shrink_factor': args.distractor_img_shrink_factor
        }
        env = DistractorWrapper(env, copy(distractor_kwargs))
        # TODO: Initialize seed for distractor

    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)

    # the dmc2gym wrapper standardizes actions
    if isinstance(env.action_space, gym.spaces.box.Box):
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
    
    return env

def make_domain_callback(args):
    if args.domain_name == 'carla':
        domain_callback = None
    elif args.domain_name == 'gridworld':
        domain_callback = None
        # domain_callback = GridWorldEvalCallback()
    else:
        domain_callback = DMCCallback({
            'domain_name': args.domain_name,
            'task_name': args.task_name,
        })
    
    return domain_callback