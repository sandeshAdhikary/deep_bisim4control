# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import argparse
import os
import gym
import time


from src.utils import utils

from types import SimpleNamespace
import os
from copy import copy
from tqdm import trange

from src.train.make_agent import make_agent
from src.train.setup_env import setup_env
from src.train.setup_logger import setup_logger
from copy import deepcopy
from einops import rearrange
from tqdm import tqdm

from src.callbacks import TrainingCallback
from src.envs.vec_wrappers import VecEnvWrapper

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default=None, type=str, 
                        help="Path to hyperparams config file. All other args are ignored when config is provided")
    # random seeds
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--num_seeds', default=1, type=int)
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=88, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none', 'mnist', 'driving_stereo'])
    parser.add_argument('--total_frames', default=1000, type=int)
    parser.add_argument('--distractor', default='None', choices=['ideal_gas', "None"])
    parser.add_argument('--distractor_type', default='None', choices=['overlay', 'padding'])
    parser.add_argument('--distractor_img_shrink_factor', default=1.3, type=float)
    parser.add_argument('--distraction_level', default=0.2, type=float)
    parser.add_argument('--boxed_env', default=False, action='store_true')
    parser.add_argument('--episode_length', default=1_000, type=int)
    # replay buffer
    parser.add_argument('--replay_buffer_capacity', default=10_000, type=int)
    # train
    parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp', 'bisim_decomp'])
    parser.add_argument('--init_steps', default=1000, type=int)
    parser.add_argument('--num_train_steps', default=1000000, type=int)
    parser.add_argument('--batch_size', default=512, type=int)
    parser.add_argument('--hidden_dim', default=256, type=int)
    parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
    parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
    parser.add_argument('--load_encoder', default=None, type=str)
    parser.add_argument('--num_train_envs', default=1, type=int)
    # eval
    parser.add_argument('--num_eval_envs', default=1, type=int)
    parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
    parser.add_argument('--num_eval_episodes', default=20, type=int)
    # critic
    parser.add_argument('--critic_lr', default=1e-3, type=float)
    parser.add_argument('--critic_beta', default=0.9, type=float)
    parser.add_argument('--critic_tau', default=0.005, type=float)
    parser.add_argument('--critic_target_update_freq', default=2, type=int)
    parser.add_argument('--use_cagrad', default=False, action='store_true')
    parser.add_argument('--vec_reward_from_model', default=False, action='store_true')
    parser.add_argument('--reward_decomp_method', default='eigenrewards', choices=['eigenrewards', 'cluster'])
    # actor
    parser.add_argument('--actor_lr', default=1e-3, type=float)
    parser.add_argument('--actor_beta', default=0.9, type=float)
    parser.add_argument('--actor_log_std_min', default=-10, type=float)
    parser.add_argument('--actor_log_std_max', default=2, type=float)
    parser.add_argument('--actor_update_freq', default=2, type=int)
    # encoder/decoder
    parser.add_argument('--encoder_type', default='pixel', type=str, choices=[
        'pixel', 'pixelCarla096', 'pixelCarla098', 'identity', 'vector',
        'pixel_cluster', 'pixelCarla096_cluster', 'pixelCarla098_cluster', 'identity_cluster', 'vector_cluster'
        ])
    parser.add_argument('--encoder_feature_dim', default=50, type=int)
    parser.add_argument('--encoder_output_dim', default=None, type=int)
    parser.add_argument('--encoder_lr', default=1e-3, type=float)
    parser.add_argument('--encoder_tau', default=0.005, type=float)
    parser.add_argument('--encoder_stride', default=1, type=int)
    parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
    parser.add_argument('--decoder_lr', default=1e-3, type=float)
    parser.add_argument('--decoder_update_freq', default=1, type=int)
    parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
    parser.add_argument('--num_layers', default=4, type=int)
    parser.add_argument('--num_filters', default=32, type=int)
    parser.add_argument('--encoder_mode', default='spectral', choices=['spectral', 'dbc'])
    parser.add_argument('--encoder_kernel_bandwidth', default='auto')
    parser.add_argument('--encoder_normalize_loss', default=True, action='store_true')
    parser.add_argument('--encoder_ortho_loss_reg', default=1e-4, type=float)
    parser.add_argument('--reward_decoder_num_rews', default=1, type=int)
    # sac
    parser.add_argument('--discount', default=0.99, type=float)
    parser.add_argument('--init_temperature', default=0.01, type=float)
    parser.add_argument('--alpha_lr', default=1e-3, type=float)
    parser.add_argument('--alpha_beta', default=0.9, type=float)
    # misc
    parser.add_argument('--work_dir', default='workdir', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    # Evaluation
    parser.add_argument('--eval_img_sources', default=None, type=str)
    # Logging
    parser.add_argument('--logger', default='tensorboard', type=str, choices=['tensorboard', 'wandb'])
    parser.add_argument('--logger_project', default='misc', type=str)
    parser.add_argument('--log_dir', default='/project/logdir', type=str)
    parser.add_argument('--logger_img_downscale_factor', default=3, type=int)
    parser.add_argument('--logger_video_log_freq', default=None, type=int)
    parser.add_argument('--logger_tags', default=None, type=str)
    parser.add_argument('--logger_minimal', default=False, action='store_true')
    # level set experiment args
    parser.add_argument('--levelset_factor', default=1.0, type=float)
    # Sweep config
    parser.add_argument('--sweep_config', default=None, type=str)
    args = parser.parse_args()
    
    if args.config is not None:
        args = args_from_config(args.config)

    return args


def args_from_config(config_path):

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config.get('project_name') is not None:
        config['logger_project'] = config.pop('project_name')
    else:
        assert config['logger_project'] is not None

    args = SimpleNamespace(**config)

    return args

def update_args(args):

    
    # Convert eval_img_sources from a single list to a list of strings
    if isinstance(args.eval_img_sources, str):
        args.eval_img_sources = args.eval_img_sources.strip("(')").replace("'", "")
        args.eval_img_sources = args.eval_img_sources.replace("[", "")
        args.eval_img_sources = args.eval_img_sources.replace("]", "")
        args.eval_img_sources = [item.strip() for item in args.eval_img_sources.split(',')]


    if hasattr(args, 'logger_tags') and isinstance(args.logger_tags, str):
        args.logger_tags = args.logger_tags.strip("(')").replace("'", "")
        args.logger_tags = args.logger_tags.replace("[", "")
        args.logger_tags = args.logger_tags.replace("]", "")
        args.logger_tags = [item.strip() for item in args.logger_tags.split(',')]


    # Set encoder output dim to be feature dim if not set
    if not isinstance(args.encoder_output_dim, int):
        args.encoder_output_dim = args.encoder_feature_dim

    # If using cluster encoders, num_rews is the same as encoder's output dim
    if 'cluster' in args.encoder_type:
        args.reward_decoder_num_rews = args.encoder_output_dim

    if args.img_source == 'none':
        args.img_source = None

    # Set logger_video_log_freq
    if args.logger_video_log_freq in [None, 'none', 'None']:
        # Set logger_video_log_freq so we get max 5 videos per run
        num_video_logs = 5
        num_evals = int(args.num_train_steps // args.eval_freq)
        args.logger_video_log_freq = max(int(num_evals / num_video_logs), 1)

    return args

def train_step():
    pass

def after_train_step(step, episode, episode_reward, agent, env, args, replay_buffer, L, training_callback):
    if args.decoder_type == 'inverse':
        raise NotImplementedError("Need to implement k_obses for inverse model")
        # for i in range(1, args.k):  # fill k_obs with 0s if episode is done
        #     replay_buffer.k_obses[replay_buffer.idx - i] = 0
    if step > 0:
        L.log('train/duration', time.time() - start_time, step)
        start_time = time.time()
        L.dump(step)

    if step % args.eval_freq == 0 and step > 0:
        L.log('eval/episode', episode, step)
        training_callback(agent, L, step)

    L.log('train/episode_reward', episode_reward, step)

    obs = env.reset()
    if len(obs) == 2:
        # env.reset() can output (obs,info) tuple
        assert isinstance(obs[1], dict)
        obs = obs[0]


    L.log('train/episode', episode, step)



def single_run(args):

    # Create workdir
    if not os.path.exists(args.work_dir):
        os.makedirs(args.work_dir)
    
    # Create project log dir
    project_log_dir = os.path.join(args.log_dir, args.logger_project)
    if not os.path.exists(project_log_dir):
        os.makedirs(project_log_dir)


    # Set seed for reproducibility
    utils.set_seed_everywhere(args.seed)

    # Setup environments and domain callbacks
    env, eval_env, domain_callback = setup_env(args)

    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    if isinstance(env, VecEnvWrapper):
        obs_shape = env.base_observation_space
        action_shape = env.base_action_space
    elif isinstance(env, (gym.vector.AsyncVectorEnv, gym.vector.SyncVectorEnv, gym.vector.VectorEnv)):
        obs_shape = obs_shape[1:] # skip the num_envs dimension
        action_shape = action_shape[1:] # skip the num_envs dimension
    else:
        ValueError("The env must be wrapped in a VecEnvWrapper")
    


    # Add existing callbacks to TrainingCallback
    minimal_logging = args.logger_minimal
    sub_callbacks = [] if minimal_logging else [domain_callback]
    training_callback = TrainingCallback({
        'work_dir': args.work_dir, 
        'save_video': args.save_video,
        'train_args': args,
        'save_model_mode': 'best',
        'num_eval_episodes': args.num_eval_episodes,
        'callbacks': sub_callbacks,
        'log_video_freq': args.logger_video_log_freq,
        'sweep_config': args.sweep_config,
    })
    training_callback.set_env(eval_env) # Will also set env for domain_callback

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')    
    # Set up the replay buffer
    replay_buffer = utils.ReplayBuffer(
        obs_shape=obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        store_infos=args.agent.endswith('decomp')
    )
    

    # Create the agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device
    )

    # Set up logger
    L = setup_logger(args)
    

    # episode, episode_reward, done = 0, 0, True


    start_time = time.time()
    prog_bar = trange(0, args.num_train_steps, env.num_envs, desc="Training")
    obs_snaps = [] # For logging first 5 training observations

    # Initialize
    obs, info = env.reset()
    episode = np.zeros(env.num_envs)
    episode_reward = np.zeros(env.num_envs)
    episode_reward_list = [[] for _ in range(env.num_envs)]
    done = [False]*env.num_envs
    reward = [0]*env.num_envs
    num_model_updates = 0

    for step in prog_bar:
        ep_done = any(done)
        if ep_done:
            # Note: the env is not explicitly reset since VecEnv resets automatically 
            #       once an environment is done

            # One of the environments is done
            episode += done 

            # Update episode rewards
            for idx in range(env.num_envs):
                if done[idx]:
                    # Record episode reward in list
                    episode_reward_list[idx].append(episode_reward[idx])
                    # Clear current episode reward tracker if done
                    episode_reward[idx] = 0
                    
        train_end_exception = None
        if step > 0 and (step % args.eval_freq == 0):
            # Only log if all environments have completed at least one episode
            if all([x >= 1 for x in episode]):
                # avg_ep_reward is the average of last episode rewards for all envs
                avg_ep_reward = np.mean([x[-1] for x in episode_reward_list])
                L.log('train/episode_reward', avg_ep_reward, step)


            # Call the training_callback
            train_end_exception = training_callback(agent, L, step)
            if train_end_exception is not None:
                avg_ep_reward = training_callback.after_train(agent, L, step)
                L.finish()
                return avg_ep_reward, train_end_exception


        # Number of episodes completed is the sum from all environments
        total_eps = sum(episode)
        L.log('train/episode', total_eps, step)
        L.log('train/num_model_updates', num_model_updates, step)


        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs, batched=True) # batched since using multi_env

        # run training update
        if step >= args.init_steps:
            # As many updates as there are environments
            num_updates = args.init_steps if step == args.init_steps else env.num_envs
            for idu in range(num_updates):
                agent.update(replay_buffer, L, step)
                num_model_updates += 1
                prog_bar_message = f"Training: Update: {idu+1}/{num_updates}"
                prog_bar.set_description(prog_bar_message)

        prog_bar_message = f"Step: {step}/{args.num_train_steps}"
        prog_bar.set_description(prog_bar_message)

        curr_reward = reward
        next_obs, reward, terminated, truncated, info = env.step(action)
        done = [x or y for (x,y) in zip(terminated, truncated)]
        
        # Allow infinite bootstrap: In the buffer, don't store truncated as done
        buffer_sample = obs, action, curr_reward, reward, next_obs, terminated

        # Take a snapshot of first 5 training observations for inspection
        if not minimal_logging:
            if step < 5:
                # Snapshot only from first env
                obs_snap = rearrange(obs[0,:], '(f c) h w -> c h (f w)', f=args.frame_stack)
                obs_snaps.append(obs_snap)
            if step == 5:
                L.log_video('train/buffer_snapshot', np.stack(obs_snaps), image_mode='chw', step=step)


        if hasattr(replay_buffer, 'infos') and (replay_buffer.infos is not None):
            buffer_sample = (*buffer_sample, info)

        replay_buffer.add(*buffer_sample, batched=True) # batched because multi_env

        for ido in range(env.num_envs):
            # The buffer counter will have moved up by env.num_envs, so correct for that
            step = replay_buffer.idx - (env.num_envs-ido)
            if args.decoder_type == 'inverse':
                # TODO: If using 'inverse model', need to subtract args.k from step?
                raise NotImplementedError("Need to implement k_obses for inverse model")
                # step = step - args.k
                # np.copyto(replay_buffer.k_obses[step - args.k], next_obs[ido,:])
        
        for idx in range(env.num_envs):
            episode_reward[idx] += reward[idx].item()
    
        obs = next_obs
    
    # Evaluate after training end
    avg_ep_reward = training_callback.after_train(agent, L, step)
    
    # Close logger
    L.finish()

    return avg_ep_reward, train_end_exception

def run_train(args=None):
    """
    Run the single_run() function potentially with multiple seeds
    """

    if args is not None:
        assert isinstance(args, (SimpleNamespace, dict))
        if isinstance(args, dict):
            args = SimpleNamespace(**args)
    else:
        args = parse_args()


    # Update args if needed
    args = update_args(args)

    seed = args.seed
    seeds = [seed + idx for idx in range(args.num_seeds)]
    avg_rewards = []
    exception = None
    for seed in seeds:
        # Set seed for reproducibility
        run_args = deepcopy(args)
        run_args.seed = seed
        avg_reward, exception = single_run(run_args)
        if exception is not None:
            # Ended due to exception
            return avg_reward, exception
        avg_rewards.append(avg_reward)

    return np.nanmedian(avg_rewards), exception



if __name__ == '__main__':
    run_train()