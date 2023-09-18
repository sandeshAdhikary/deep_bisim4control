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
import json
import dmc2gym

import utils
from logger import Logger
from video import VideoRecorder
from types import SimpleNamespace

from agent.baseline_agent import BaselineAgent
from agent.bisim_agent import BisimAgent
from agent.bisim_agent_decomp import BisimAgentDecomp
from agent.deepmdp_agent import DeepMDPAgent
from tqdm import trange
from envs.reacher import make_reacher
from envs.reacher.callbacks import ReacherEvalCallback
from envs.gridworld import make_gridworld
from envs.gridworld.callbacks import GridWorldEvalCallback
from envs.dmc2gym.callbacks import DMCCallback
from envs.distractor_wrappers import DistractorWrapper
import os
from copy import copy
from einops import rearrange

def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain_name', default='cheetah')
    parser.add_argument('--task_name', default='run')
    parser.add_argument('--image_size', default=88, type=int)
    parser.add_argument('--action_repeat', default=1, type=int)
    parser.add_argument('--frame_stack', default=3, type=int)
    parser.add_argument('--resource_files', type=str)
    parser.add_argument('--eval_resource_files', type=str)
    parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none'])
    parser.add_argument('--total_frames', default=1000, type=int)
    parser.add_argument('--distractor', default='None', choices=['ideal_gas', "None"])
    parser.add_argument('--distractor_type', default='None', choices=['overlay', 'padding'])
    parser.add_argument('--distractor_img_shrink_factor', default=1.3, type=float)
    parser.add_argument('--distraction_level', default=0.2, type=float)
    parser.add_argument('--boxed_env', default=False, action='store_true')
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
    # eval
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
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--work_dir', default='.', type=str)
    parser.add_argument('--save_tb', default=False, action='store_true')
    parser.add_argument('--save_model', default=False, action='store_true')
    parser.add_argument('--save_buffer', default=False, action='store_true')
    parser.add_argument('--save_video', default=False, action='store_true')
    parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
    parser.add_argument('--render', default=False, action='store_true')
    parser.add_argument('--port', default=2000, type=int)
    parser.add_argument('--logger', default='tensorboard', type=str, choices=['tensorboard', 'wandb'])
    parser.add_argument('--logger_project', default='bisim_exp', type=str)
    args = parser.parse_args()

    return args


def evaluate(env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None):
    # carla metrics:
    reason_each_episode_ended = []
    distance_driven_each_episode = []
    crash_intensity = 0.
    steer = 0.
    brake = 0.
    count = 0
    max_obs_video_frames = 200

    # embedding visualization
    obses = []
    values = []
    embeddings = []
    episode_rewards = []
    for i in range(num_episodes):
        # carla metrics:
        dist_driven_this_episode = 0.

        obs = env.reset()
        if len(obs) == 2:
            # env.reset() can output (obs,info) tuple
            assert isinstance(obs[1], dict)
            obs = obs[0]
        video.init(enabled=(i == 0))
        done = False
        episode_reward = 0
        while not done:
            with utils.eval_mode(agent):
                action = agent.select_action(obs)
            if i == 0:
                obses.append(obs)

            if embed_viz_dir:
                with torch.no_grad():
                    values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
                    embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

            obs, reward, terminated, truncated, info = env.step(action)
            # TODO: Should done=truncated or terminated?
            done = truncated or terminated


            # metrics:
            if do_carla_metrics:
                dist_driven_this_episode += info['distance']
                crash_intensity += info['crash_intensity']
                steer += abs(info['steer'])
                brake += info['brake']
                count += 1

            video.record(env)
            episode_reward += reward

        # metrics:
        if do_carla_metrics:
            reason_each_episode_ended.append(info['reason_episode_ended'])
            distance_driven_each_episode.append(dist_driven_this_episode)

        video.save('%d.mp4' % step)
        if len(video.frames) > 0:
            L.log_video('eval/video', video.frames, step)
        L.log('eval/episode_reward', episode_reward, step)
        episode_rewards.append(episode_reward)

    if embed_viz_dir:
        dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

    obses = np.stack(obses)
    if len(obses.shape) == 4:
        # Image observations
        obses = rearrange(obses, 'b (f c) h w -> b c h (f w)', f=len(env._frames)) # Stack frames horizontally
        L.log_video('eval/obs_video', obses[:max_obs_video_frames, :], image_mode='chw', step=step)

    L.dump(step)

    if do_carla_metrics:
        print('METRICS--------------------------')
        print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        print('steer: {}'.format(steer / count))
        print('brake: {}'.format(brake / count))
        print('---------------------------------')

    return np.median(episode_rewards)

def make_agent(obs_shape, action_shape, args, device):
    if (args.agent is None) or (args.agent == np.nan):
        raise ValueError("args.agent is None or NaN")
    if args.agent == 'baseline':
        agent = BaselineAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )
    elif args.agent == 'bisim':
        agent = BisimAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef,
            encoder_kernel_bandwidth=args.encoder_kernel_bandwidth,
            encoder_normalize_loss=args.encoder_normalize_loss,
            encoder_ortho_loss_reg=args.encoder_ortho_loss_reg,
            encoder_mode=args.encoder_mode,
            reward_decoder_num_rews=args.reward_decoder_num_rews,
            encoder_output_dim=args.encoder_output_dim
        )
    elif args.agent == 'bisim_decomp':
        agent = BisimAgentDecomp(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            encoder_stride=args.encoder_stride,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            bisim_coef=args.bisim_coef,
            encoder_kernel_bandwidth=args.encoder_kernel_bandwidth,
            encoder_normalize_loss=args.encoder_normalize_loss,
            encoder_ortho_loss_reg=args.encoder_ortho_loss_reg,
            encoder_mode=args.encoder_mode,
            reward_decoder_num_rews=args.reward_decoder_num_rews,
            encoder_output_dim=args.encoder_output_dim,
            use_cagrad=args.use_cagrad,
            reward_decomp_method=args.reward_decomp_method
        )
    elif args.agent == 'deepmdp':
        agent = DeepMDPAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            encoder_stride=args.encoder_stride,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            decoder_type=args.decoder_type,
            decoder_lr=args.decoder_lr,
            decoder_update_freq=args.decoder_update_freq,
            decoder_weight_lambda=args.decoder_weight_lambda,
            transition_model_type=args.transition_model_type,
            num_layers=args.num_layers,
            num_filters=args.num_filters
        )

    if args.load_encoder:
        model_dict = agent.actor.encoder.state_dict()
        encoder_dict = torch.load(args.load_encoder) 
        encoder_dict = {k[8:]: v for k, v in encoder_dict.items() if 'encoder.' in k}  # hack to remove encoder. string
        agent.actor.encoder.load_state_dict(encoder_dict)
        agent.critic.encoder.load_state_dict(encoder_dict)

    return agent


def run_train(args=None):


    if args is not None:
        assert isinstance(args, (SimpleNamespace, dict))
        if isinstance(args, dict):
            args = SimpleNamespace(**args)
    else:
        args = parse_args()

    # Set encoder output dim to be feature dim if not set
    if not isinstance(args.encoder_output_dim, int):
        args.encoder_output_dim = args.encoder_feature_dim

    # If using cluster encoders, num_rews is the same as encoder's output dim
    if 'cluster' in args.encoder_type:
        args.reward_decoder_num_rews = args.encoder_output_dim

    utils.set_seed_everywhere(args.seed)

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
        eval_env = env
        eval_callback = None
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
            boxed_env=args.boxed_env
        )
        eval_env = make_gridworld(
            domain_name=args.domain_name,
            task_name=args.task_name,
            from_pixels=args.encoder_type.startswith('pixel'),
            seed=args.seed + 1,
            height=args.image_size,
            width=args.image_size,
            size=size,
            boxed_env=args.boxed_env
        )
        eval_callback = GridWorldEvalCallback()
    elif args.domain_name == 'reacher':
        env = make_reacher(
            domain_name=args.domain_name,
            task_name=args.task_name,
            from_pixels=args.encoder_type.startswith('pixel'),
            seed=args.seed,
            height=args.image_size,
            width=args.image_size
        )
        eval_env = make_reacher(
            domain_name=args.domain_name,
            task_name=args.task_name,
            from_pixels=args.encoder_type.startswith('pixel'),
            seed=args.seed +1,
            height=args.image_size,
            width=args.image_size
        )
        eval_callback = ReacherEvalCallback()
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
        )
        env.seed(args.seed)

        eval_env = dmc2gym.make(
            domain_name=args.domain_name,
            task_name=args.task_name,
            resource_files=args.eval_resource_files,
            img_source=args.img_source,
            total_frames=args.total_frames,
            seed=args.seed,
            visualize_reward=False,
            from_pixels=args.encoder_type.startswith('pixel'),
            height=args.image_size,
            width=args.image_size,
            frame_skip=args.action_repeat,
        )
        eval_callback = DMCCallback({
            'domain_name': args.domain_name,
            'task_name': args.task_name,
        })

    if args.distractor in ['ideal_gas']:
        distractor_kwargs = {
            'num_bodies': int(args.distraction_level*20),
            'num_dimensions': 2,
            'distractor_type': args.distractor_type,
            'img_shrink_factor': args.distractor_img_shrink_factor
        }
        env = DistractorWrapper(env, copy(distractor_kwargs))
        eval_env = DistractorWrapper(eval_env, copy(distractor_kwargs))


    # stack several consecutive frames together
    if args.encoder_type.startswith('pixel'):
        env = utils.FrameStack(env, k=args.frame_stack)
        eval_env = utils.FrameStack(eval_env, k=args.frame_stack)

    if eval_callback is not None:
        eval_callback.set_env(eval_env)

    utils.make_dir(args.work_dir)
    video_dir = utils.make_dir(os.path.join(args.work_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.work_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.work_dir, 'buffer'))

    video = VideoRecorder(video_dir if args.save_video else None)

    with open(os.path.join(args.work_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # the dmc2gym wrapper standardizes actions
    if isinstance(env.action_space, gym.spaces.box.Box):
        assert env.action_space.low.min() >= -1
        assert env.action_space.high.max() <= 1
    
    store_info_in_bueer = args.agent.endswith('decomp')
    replay_buffer = utils.ReplayBuffer(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        store_infos=store_info_in_bueer
    )
 

    agent = make_agent(
        obs_shape=env.observation_space.shape,
        action_shape=env.action_space.shape,
        args=args,
        device=device
    )

    # Set up logger
    if args.logger == 'tensorboard':
        logger_config = {'log_dir': args.work_dir,
                         'sw': 'tensorboard', 
                         'format_config': 'rl'
                         }
    elif args.logger == 'wandb':
        logger_config = {'log_dir': args.work_dir,
                         'sw': 'wandb',
                         'project': args.logger_project,
                         'tracked_params': args.__dict__
                         }
    L = Logger(logger_config)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()
    prog_bar = trange(args.num_train_steps, desc="Training")
    for step in prog_bar:
        if done:
            if args.decoder_type == 'inverse':
                for i in range(1, args.k):  # fill k_obs with 0s if episode is done
                    replay_buffer.k_obses[replay_buffer.idx - i] = 0
            if step > 0:
                L.log('train/duration', time.time() - start_time, step)
                start_time = time.time()
                L.dump(step)

            # evaluate agent periodically
            if episode % args.eval_freq == 0 and episode > 0:
                L.log('eval/episode', episode, step)
                evaluate(eval_env, agent, video, args.num_eval_episodes, L, step)
                if args.save_model:
                    agent.save(model_dir, step)
                if args.save_buffer:
                    replay_buffer.save(buffer_dir)
                if eval_callback is not None:
                    eval_callback(agent, L, step)

            L.log('train/episode_reward', episode_reward, step)

            obs = env.reset()
            if len(obs) == 2:
                # env.reset() can output (obs,info) tuple
                assert isinstance(obs[1], dict)
                obs = obs[0]
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            reward = 0

            L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = args.init_steps if step == args.init_steps else 1
            for idu in range(num_updates):
                agent.update(replay_buffer, L, step)
                prog_bar.set_description_str(f"Training: Update: {idu+1}/{num_updates}")

        curr_reward = reward
        next_obs, reward, terminated, truncated, info = env.step(action)
        # TODO: Should done be terminated or truncated?
        done = terminated or truncated
        # allow infinit bootstrap
        max_ep_steps = None
        if hasattr(env, "_max_episode_steps"):
            max_ep_steps = env._max_episode_steps
        elif hasattr(env, "max_episode_steps"):
            max_ep_steps = env.max_episode_steps
        else:
            raise ValueError("env has no attribute max_episode_steps or _max_episode_steps")

        done_bool = done
        if max_ep_steps is not None:
            done_bool = 0 if episode_step + 1 == max_ep_steps else float(
                done
            )
        episode_reward += reward
        
        buffer_sample = obs, action, curr_reward, reward, next_obs, done_bool
        if hasattr(replay_buffer, 'infos') and (replay_buffer.infos is not None):
            buffer_sample = (*buffer_sample, info)

        replay_buffer.add(*buffer_sample)
        np.copyto(replay_buffer.k_obses[replay_buffer.idx - args.k], next_obs)

        obs = next_obs
        episode_step += 1
    
    # Evaluate after training end
    avg_ep_reward = evaluate(env, agent, video, args.num_eval_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None)
    
    # Close logger
    L.finish()

    return avg_ep_reward


def collect_data(env, agent, num_rollouts, path_length, checkpoint_path):
    rollouts = []
    for i in range(num_rollouts):
        obses = []
        acs = []
        rews = []
        observation = env.reset()
        for j in range(path_length):
            action = agent.sample_action(observation)
            next_observation, reward, done, _ = env.step(action)
            obses.append(observation)
            acs.append(action)
            rews.append(reward)
            observation = next_observation
        obses.append(next_observation)
        rollouts.append((obses, acs, rews))

    from scipy.io import savemat

    savemat(
        os.path.join(checkpoint_path, "dynamics-data.mat"),
        {
            "trajs": np.array([path[0] for path in rollouts]),
            "acs": np.array([path[1] for path in rollouts]),
            "rews": np.array([path[2] for path in rollouts])
        }
    )


if __name__ == '__main__':
    run_train()
