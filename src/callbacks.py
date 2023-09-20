import utils
import torch
import numpy as np
import os
from einops import rearrange
import matplotlib.pyplot as plt
from video import VideoRecorder
import json
import wandb


class TrainingCallback():
    
    def __init__(self, config=None):
        config = config or {}

        # Set up working dir and save args
        self.work_dir = config.get('work_dir', '.')
        self.model_dir = utils.make_dir(os.path.join(self.work_dir, 'model'))
        self.buffer_dir = utils.make_dir(os.path.join(self.work_dir, 'buffer'))
        # Save snapshot of input args
        self.train_args = config.get('train_args', None)
        utils.make_dir(self.work_dir)
        if self.train_args is not None:
            # If train_args provided, save them as json
            with open(os.path.join(self.work_dir, 'args.json'), 'w') as f:
                json.dump(vars(self.train_args), f, sort_keys=True, indent=4)

        # Model saving:
        self.save_model_mode = config.get('save_model_mode', None)
        assert self.save_model_mode in [None, 'best', 'last', 'all']
        self.best_eval_reward = -np.inf

        # Set up video recorder
        save_video = config.get('save_video', False)
        video_dir = utils.make_dir(os.path.join(self.work_dir, 'video'))
        self.video = VideoRecorder(video_dir if save_video else None)
        self.num_eval_episodes = config.get('num_eval_episodes', 1)
        
        # Set up performance logging
        self.eval_table = None

        # Add other callbacks
        self.callbacks = config.get('callbacks')

    def set_env(self, env):
        self.env = env
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                if callback is not None:
                    callback.set_env(env)

    def __call__(self, agent, logger, step):
        eval_reward = self.evaluate(self.env, agent, self.video, self.num_eval_episodes, logger, step)
        if self.save_model_mode == 'all':
            # save model
            agent.save(self.model_dir, step)
            # logger.log_agent("agent", agent, step)
        elif self.save_model_mode == 'best' and eval_reward >= self.best_eval_reward:
            # save model
            agent.save(self.model_dir, f"step_{step}_best")
            # logger.log_agent("best_agent", agent, step)

        if self.callbacks is not None:
            for callback in self.callbacks:
                if callback is not None:
                    callback.__call__(agent, logger, step)
        logger.dump(step)
    
    def after_train(self, agent, logger, step):
        """
        """
        # Evaluate the model and get final performance metric
        eval_reward = self.evaluate(self.env, agent, self.video, self.num_eval_episodes, logger, step)
        
        if self.callbacks is not None:
            for callback in self.callbacks:
                if (callback is not None) and hasattr(callback, 'after_train'):
                    callback.after_train(agent, logger, step)  
        
        logger.dump(step)

        return eval_reward


    def evaluate(self, env, agent, video, num_episodes, L, step, device=None, embed_viz_dir=None, do_carla_metrics=None):
        # carla metrics:
        reason_each_episode_ended = []
        distance_driven_each_episode = []
        crash_intensity = 0.
        steer = 0.
        brake = 0.
        count = 0

        # embedding visualization
        obses = []
        values = []
        embeddings = []
        episode_rewards = []
        episode_reward_lists = []
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
            episode_reward_list = []
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
                episode_reward_list.append(reward)
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
            episode_reward_lists.append(episode_reward_list)

        # Get mean reward over episodes
        mean_ep_reward = np.median(episode_rewards)
        self.log_performance(episode_rewards, L, step)

        if embed_viz_dir:
            dataset = {'obs': obses, 'values': values, 'embeddings': embeddings}
            torch.save(dataset, os.path.join(embed_viz_dir, 'train_dataset_{}.pt'.format(step)))

        obses = np.stack(obses)
        if len(obses.shape) == 4:
            # Image observations
            obses = rearrange(obses, 'b (f c) h w -> b c h (f w)', f=len(env._frames)) # Stack frames horizontally
            L.log_video('eval/obs_video', obses, image_mode='chw', step=step)

        # Plot reward trajectory
        episode_reward_lists = np.array(episode_reward_lists)
        fig, ax = plt.subplots(1,1)
        [ax.plot(episode_reward_lists[idx], '-o', label=f'Episode {idx}') for idx in range(episode_reward_lists.shape[0])]
        plt.legend()
        rewards_traj = utils.plot_to_array(fig)
        L.log_image('eval/reward_trajectory', rewards_traj, step=step)
        plt.close()

        if do_carla_metrics:
            print('METRICS--------------------------')
            print("reason_each_episode_ended: {}".format(reason_each_episode_ended))
            print("distance_driven_each_episode: {}".format(distance_driven_each_episode))
            print('crash_intensity: {}'.format(crash_intensity / num_episodes))
            print('steer: {}'.format(steer / count))
            print('brake: {}'.format(brake / count))
            print('---------------------------------')

        return mean_ep_reward

    def log_performance(self, reward_list, logger, step):
        avg_reward = np.median(reward_list)
        std_reward = np.std(reward_list)
        # std_reward = np.random.uniform()

        # if self.eval_table is None:
        if self.eval_table is None:
            eval_data = [[step, avg_reward, std_reward]]
            columns = ['step', 'value', 'error']
        else:
            eval_data = self.eval_table.data    
            columns = self.eval_table.columns
            eval_data.append([step, avg_reward, std_reward])
            
        self.eval_table = wandb.Table(data=eval_data, columns=columns)

        wandb.log(
            {
                "eval_rewards_lineplot": wandb.plot_table(
                    data_table=self.eval_table ,
                    vega_spec_name='adhikary-sandesh/lineploterrorband',
                    fields={'value': 'value', 'error': 'error', 'step': 'step'},
                )
            }
        )
