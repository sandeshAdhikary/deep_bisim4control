from src.utils import utils
import torch
import numpy as np
import os
from einops import rearrange
import matplotlib.pyplot as plt
from src.utils.video import VideoRecorder
import json
import wandb
from tqdm import trange
import optuna
from copy import deepcopy
from sqlalchemy import create_engine, Table, Column, Integer, String, MetaData, insert, values, text, JSON
from optuna.pruners._successive_halving import (
    _get_current_rung, _completed_rung_key, _get_competing_values,
    _estimate_min_resource, _is_trial_promotable_to_next_rung
)
import pytz
import datetime
from optuna.pruners import SuccessiveHalvingPruner
tz = pytz.timezone('America/Los_Angeles')
from gym.vector import AsyncVectorEnv, SyncVectorEnv, VectorEnv, VectorEnvWrapper
VEC_ENVS = (AsyncVectorEnv, SyncVectorEnv, VectorEnv, VectorEnvWrapper)
from src.defaults import DEFAULTS
DEFAULT_OPTUNA_STORAGE = DEFAULTS['OPTUNA_STORAGE']

class TrainingCallback():
    
    def __init__(self, config=None):
        config = config or {}

        # Set up working dir and save args
        self.log_dir = config.get('logdir', '.')
        self.model_dir = utils.make_dir(os.path.join(self.log_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(self.log_dir, 'buffer'))
        # Save snapshot of input args
        self.train_args = config.get('train_args', None)
        utils.make_dir(self.log_dir)
        if self.train_args is not None:
            # If train_args provided, save them as json
            with open(os.path.join(self.log_dir, 'args.json'), 'w') as f:
                args_to_save = deepcopy(vars(self.train_args))
                args_to_save['sweep_config'] = None # Don't save since it is not serializable
                args_to_save['sweep_info'] = None # Don't save since it is not serializable
                json.dump(args_to_save, f, sort_keys=True, indent=4)

        # Model saving:
        self.save_model_mode = config.get('save_model_mode', None)
        if self.save_model_mode not in [None, 'none', 'None', 'all']:
            # Other option is "every_<int>"
            save_every = self.save_model_mode.split("_")
            assert len(save_every) == 2 and save_every[0] == 'every' and save_every[1].isdigit()
        # Note: the best model is always saved
        self.best_eval_reward = -np.inf

        # Set up video recorder
        save_video = config.get('save_video', False)
        video_dir = utils.make_dir(os.path.join(self.log_dir, 'video'))
        self.video = VideoRecorder(video_dir if save_video else None)
        self.num_eval_episodes = config.get('num_eval_episodes', 1)
        
        # Set up performance logging
        self.eval_table = None

        # Add other callbacks
        self.callbacks = config.get('callbacks')
        self.log_video_freq = config.get('log_video_freq', 10) # Every 10 evals
        self.eval_calls = 0 # Track how many times eval has been called
        self.training_done = False

        self.sweep_callback = None
        sweep_config = config.get('sweep_config')
        if sweep_config:
            self.sweep_callback = SweepCallback(sweep_config)

        self.train_end_exception = None

    def set_env(self, env):
        self.env = env
        
        self.vectorized = False
        if hasattr(self.env, 'envs'):
            # Use functions for vectorized environments
            self.vectorized = True
    
        if self.callbacks is not None:
            for callback in self.callbacks:
                if callback is not None:
                    callback.set_env(env)

    def __call__(self, agent, logger, step):

        eval_reward = self.evaluate(self.env, agent, self.video, self.num_eval_episodes, logger, step)
        if self.save_model_mode == 'all':
            logger.log_agent('train/agent', agent, step)
        elif self.save_model_mode.startswith('every'):
            
            if step % int(self.save_model_mode.split("_")[1]) == 0:
                logger.log_agent('train/agent', agent, step)
        

        # Save the best model
        if eval_reward >= self.best_eval_reward:
            # save model
            agent.save(self.model_dir, f"step_{step}_best")
            self.best_eval_reward = eval_reward

        if self.callbacks is not None:
            for callback in self.callbacks:
                if callback is not None:
                    callback.__call__(agent, logger, step)


        if self.sweep_callback is not None:
            info = {'eval_reward': eval_reward}
            exception = self.sweep_callback(agent, logger, info, step)
            if exception is not None:
                return exception

        logger.dump(step)


        self.eval_calls += 1

    def after_train(self, agent, logger, step):
        """
        """
        # Evaluate the model and get final performance metric
        self.training_done = True
        eval_reward = self.evaluate(self.env, agent, self.video, self.num_eval_episodes, logger, step)
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                if (callback is not None) and hasattr(callback, 'after_train'):
                    callback.after_train(agent, logger, step)  
        
        logger.dump(step)

        return eval_reward
    
    def evaluate(self, env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None):
        if self.vectorized:
            return self.evaluate_vec(env, agent, video, num_episodes, L, step, device, embed_viz_dir, do_carla_metrics)
        else:
            return self.evaluate_nonvec(env, agent, video, num_episodes, L, step, device, embed_viz_dir, do_carla_metrics)


    def evaluate_vec(self, env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None):
        max_ep_steps = env._max_episode_steps
        num_frames = env._frames

        obses = []       
        video.init(enabled=True)

        obs, info = env.reset() # obs: (num_env, *obs_shape)
        dones = [False] * env.num_envs
        episode_reward_list = []
        steps = 0
        steps_to_keep = [None]*env.num_envs
        num_eps = [0]*env.num_envs

        for i in trange(num_episodes*max_ep_steps, desc='Running eval rollouts'):
            # No  need to reset again since vec_env resets individual envs automatically
            if None not in steps_to_keep:
                # All envs have completed max_eps
                break
            steps += 1
            with utils.eval_mode(agent):
                # Get actions for all envs
                action = agent.select_action(obs, batched=True)

            obses.append(obs)

            obs, reward, terminated, truncated, info = env.step(action)
            dones = [x or y for (x,y) in zip(terminated, truncated)]
    
            for ide in range(env.num_envs):
                if dones[ide]:
                    num_eps[ide] += 1

                if num_eps[ide] >= num_episodes:
                    steps_to_keep[ide] = steps
            if (self.eval_calls % self.log_video_freq == 0) or self.training_done:
                video.record(env)
            episode_reward_list.append(reward)

        if (self.eval_calls % self.log_video_freq == 0) or self.training_done:
            video.save('%d.mp4' % step)
            if len(video.frames) > 0:
                L.log_video('eval/video', video.frames, step)
        episode_reward_list = np.array(episode_reward_list)

        max_steps = episode_reward_list.shape[0]
        mask = [np.pad(np.ones(n), (0, max_steps-n),mode='constant') for n in steps_to_keep]
        mask = np.stack(mask, axis=1)
        episode_reward_list *= mask

        # Log video of evaluation observations
        obses = np.stack(obses) # (steps, num_envs, *obs_shape)
        # Stack frames horizontally; Stack episodes along batches
        obses = rearrange(obses, 'b n (f c) h w -> b c (n h) (f w)', f=len(num_frames)) 
        if (self.eval_calls % self.log_video_freq == 0) or self.training_done:
            L.log_video('eval/obs_video', obses, image_mode='chw', step=step)


        # Get average episode reward for each environment
        avg_env_episode_rewards = np.sum(episode_reward_list, axis=0)/num_episodes
        # Log performance for each environment
        self.log_performance(avg_env_episode_rewards, L, step)


        # eval_episode_reward is the episode reward only from the first env
        # since first env has the same config as the training env
        eval_episode_reward = avg_env_episode_rewards[0]

        L.log('eval/episode_reward', eval_episode_reward, step)

        return eval_episode_reward


    def evaluate_nonvec(self, env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None):
        raise NotImplementedError
        # # carla metrics:
        # reason_each_episode_ended = []
        # distance_driven_each_episode = []
        # crash_intensity = 0.
        # steer = 0.
        # brake = 0.
        # count = 0

        # # embedding visualization
        # obses = []
        # values = []
        # embeddings = []
        # episode_rewards = []
        # episode_reward_lists = []
        # for i in range(num_episodes):
        #     # carla metrics:
        #     dist_driven_this_episode = 0.

        #     obs, info = env.reset()
        #     video.init(enabled=(i == 0))
        #     done = False
        #     episode_reward = 0
        #     episode_reward_list = []
        #     while not done:
        #         with utils.eval_mode(agent):
        #             action = agent.select_action(obs)
        #         if i == 0:
        #             obses.append(obs)

        #         if embed_viz_dir:
        #             with torch.no_grad():
        #                 values.append(min(agent.critic(torch.Tensor(obs).to(device).unsqueeze(0), torch.Tensor(action).to(device).unsqueeze(0))).item())
        #                 embeddings.append(agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device)).cpu().detach().numpy())

        #         obs, reward, terminated, truncated, info = env.step(action)
        #         # TODO: Should done=truncated or terminated?
        #         done = truncated or terminated


        #         # metrics:
        #         if do_carla_metrics:
        #             dist_driven_this_episode += info['distance']
        #             crash_intensity += info['crash_intensity']
        #             steer += abs(info['steer'])
        #             brake += info['brake']
        #             count += 1

        #         video.record(env)
        #         episode_reward_list.append(reward)
        #         episode_reward += reward

        #     # metrics:
        #     if do_carla_metrics:
        #         reason_each_episode_ended.append(info['reason_episode_ended'])
        #         distance_driven_each_episode.append(dist_driven_this_episode)

        #     video.save('%d.mp4' % step)
        #     if len(video.frames) > 0:
        #         L.log_video('eval/video', video.frames, step)
        #     L.log('eval/episode_reward', episode_reward, step)
        #     episode_rewards.append(episode_reward)
        #     episode_reward_lists.append(episode_reward_list)



        # if embed_viz_dir:
        #     dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
        #     torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

        # obses = np.stack(obses)
        # if len(obses.shape) == 4:
        #     # Image observations
        #     obses = rearrange(obses, 'b (f c) h w -> b c h (f w)', f=len(env._frames)) # Stack frames horizontally
        #     L.log_video('eval/obs_video', obses, image_mode='chw', step=step)

        # # Plot reward trajectory
        # episode_reward_lists = np.array(episode_reward_lists)
        # fig, ax = plt.subplots(1,1)
        # [ax.plot(episode_reward_lists[idx], '-o', label=f'Episode {idx}') for idx in range(episode_reward_lists.shape[0])]
        # plt.legend()
        # rewards_traj = utils.plot_to_array(fig)
        # L.log_image('eval/reward_trajectory', rewards_traj, step=step)
        # plt.close()

        # if do_carla_metrics:
        #     print('METRICS--------------------------')
        #     print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
        #     print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
        #     print('crash_intensity: {}'.format(crash_intensity / num_episodes))
        #     print('steer: {}'.format(steer / count))
        #     print('brake: {}'.format(brake / count))
        #     print('---------------------------------')

        # # Get mean reward over episodes
        # mean_ep_reward = np.median(episode_rewards)
        # std_ep_reward = np.std(episode_rewards)
        # self.log_performance(episode_rewards, L, step)

        # return mean_ep_reward

    def log_performance(self, episode_rewards, logger, step):

        if logger.sw_type == 'wandb':
            
            env_cols = [f'env_{i}' for i in range(len(episode_rewards))]
            columns = ['step', 'value', 'error', *env_cols]
            avg_value = np.median(episode_rewards)
            std_value = np.std(episode_rewards)
            # if self.eval_table is None:
            if self.eval_table is None:
                eval_data = [[step, avg_value, std_value, *episode_rewards]]
            else:
                eval_data = self.eval_table.data    
                columns = self.eval_table.columns
                eval_data.append([step, avg_value, std_value, *episode_rewards])
                
            self.eval_table = wandb.Table(data=eval_data, columns=columns)
            logger._sw.log(
                {
                    "eval_rewards_lineplot": wandb.plot_table(
                        data_table=self.eval_table ,
                        vega_spec_name='adhikary-sandesh/lineploterrorband',
                        fields={'value': 'value', 'error': 'error', 'step': 'step'},
                    )
                },
                step=step
            )

class SweepCallback():
    
    def __init__(self, config=None):
        config = config or {}
        self.sweep_name = config['sweep_name']
        self.study_name = config['study_name']
        self.optuna_trial = config.get('optuna_trial')
        self.optuna_storage = config.get('optuna_storage')
        if self.optuna_trial is not None:
            self.study_id = self.optuna_trial.study._study_id
            self.sweep_id = self._get_sweep_id(self.sweep_name)


        
    def __call__(self, agent, logger, info, step):

        exception = None
        if self.optuna_trial is not None:
            exception = self._optuna_callback(self.optuna_trial, info, step)
        
        return exception


    def _get_sweep_id(self, sweep_name):
        engine = create_engine(self.optuna_storage)
        with engine.connect() as conn:
            sweep_id = conn.execute(text(f"""SELECT * FROM sweeps 
                                         WHERE sweep_name=:sweep_name 
                                         AND study_id=:study_id"""), 
                                         [{'sweep_name': sweep_name, 'study_id': self.study_id}]).fetchall()[0][0]
            # [0][0]

        return sweep_id

    def _optuna_callback(self, trial, info, step):
        # Record intermediate values
        trial.report(info['eval_reward'], step)
        
        # Record intermediate values for sweep
        current_time = datetime.datetime.now(tz).strftime("%Y-%m-%d-%H:%M:%S.%f")
        sweep_info = {'current_time': current_time, 'intermediate_value': info['eval_reward']}
        
        # Optionally, get pruner info
        pruner_info = None
        if hasattr(self.optuna_trial.study, 'pruner'):
            pruner = self.optuna_trial.study.pruner
            if isinstance(pruner, SuccessiveHalvingPruner):
                rung = _get_current_rung(self.optuna_trial)
                pruner_info = {'rung': rung}
        sweep_info['pruner_info'] = pruner_info

        sweep_info = json.dumps(sweep_info)
        engine = create_engine(self.optuna_storage)
        with engine.connect() as conn:
            stmnt = f"""
            INSERT INTO sweeps_intermediate_values (step, study_id, sweep_id, trial_id, info) 
            VALUES (:step, :study_id, :sweep_id, :trial_id, :info)
            ON DUPLICATE KEY
            UPDATE step=step, study_id=study_id, sweep_id=sweep_id, trial_id=trial_id, info=info
            """
            conn.execute(text(stmnt), [{
                'step': step,
                'sweep_id': self.sweep_id,
                'study_id': trial.study._study_id,
                'trial_id': trial._trial_id,
                'info': json.dumps(sweep_info)
            }])
            conn.commit()


            if trial.should_prune():
                return optuna.TrialPruned()
    



