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
from src.train.load_args import update_args, parse_args



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
    
    # Create project log dir
    # project_log_dir = os.path.join(args.log_dir, args.logger_project)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)


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
        'project_name': args.logger_project,
        'save_video': args.save_video,
        'train_args': args,
        'save_model_mode': args.save_model_mode,
        'num_eval_episodes': args.num_eval_episodes,
        'callbacks': sub_callbacks,
        'log_video_freq': args.logger_video_log_freq,
        'sweep_config': args.sweep_config,
        'logdir': args.log_dir
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
    
    args.agent_load_path = os.path.join(args.log_dir, 'models')
    # Create the agent
    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
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


            # Call the training_callbacks
            train_end_exception = training_callback(agent, L, step)
            if train_end_exception is not None:
                avg_ep_reward = training_callback.after_train(agent, L, step)
                if not args.retain_logger:
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
    
    if not args.retain_logger:
        # Close logger
        L.finish()

    return avg_ep_reward, train_end_exception

def run_train(args=None):
    """
    Run the single_run() function potentially with multiple seeds
    """
    if args is not None:
        args = SimpleNamespace(**args)
    else:
        args = parse_args()


    # Update args if needed
    args = update_args(args)

    seed = args.seed

    if args.num_seeds == 1:
        avg_reward, exception = single_run(args)
        return avg_reward, exception
    else:
        seeds = [seed + idx for idx in range(args.num_seeds)]
        avg_rewards = []
        exception = None
        for seed in seeds:
            # Set seed for reproducibility
            run_args = deepcopy(args)
            run_args.seed = seed
            run_args.log_dir = os.path.join(args.log_dir, f'seed_{seed}')
            avg_reward, exception = single_run(run_args)
            if exception is not None:
                # Ended due to exception
                return avg_reward, exception
            avg_rewards.append(avg_reward)

        return np.nanmedian(avg_rewards), exception



if __name__ == '__main__':
    run_train()