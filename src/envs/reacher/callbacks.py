import torch 
import numpy as np
from envs.reacher.reacher import MOReacher, MOReacherRGB
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
import matplotlib.pyplot as plt
from utils import plot_to_array
from tqdm import trange

class ReacherEvalCallback():

    def __init__(self, config=None):
        self.env = None
        config = config or {}

    def set_env(self, env):
        self.env = env

    def __call__(self, agent, logger, step):
        if agent.encoder_mode == 'spectral':
            self._spectral_call(agent, logger, step)

    def _spectral_call(self, agent, logger, step):

        if agent.reward_decoder_centroids is None:
            return None
        
        assert self.env is not None, "Environment not set!"
        num_eval_episodes = 1
        max_episode_steps = self.env.max_episode_steps

        # Collect rollouts
        rollout = self.collect_rollout(agent, num_eval_episodes, max_episode_steps)

        # Get features/embeddings for all observations
        obs = torch.from_numpy(rollout['obs']).to(agent.device)
        obs = obs.contiguous().to(torch.float32)
        obs = rearrange(obs, 'ep n c h w -> (ep n) c h w')
        features = agent.critic.encoder(obs) # (Ep*N, d)

        # Get cluster labels
        cluster_labels = torch.cdist(features, agent.reward_decoder_centroids, p=2)
        cluster_labels = torch.exp(-cluster_labels**2) # (Ep*N, d)


        # Get gradients wrt sub-rewards
        imgs = rearrange(rollout['imgs'], 'ep n h w c -> (ep n) h w c', ep=num_eval_episodes)
        

        # Plot cluster labels
        num_labels = cluster_labels.shape[1]
        width, height = 480, 480
        composite_imgs = []
        for idx in trange(cluster_labels.shape[0], desc='Plotting cluster labels'):
            fig, ax = plt.subplots(1, 1)
            ax.set_xlim(0, cluster_labels.shape[0])
            ax.set_ylim(0, 1)
            for ida in range(num_labels):
                ax.plot(range(idx), cluster_labels[:idx,ida].detach().cpu(), label=f'Cluster {ida}')
            plt.legend()
            # Get image from figure
            fig_arr = plot_to_array(fig) # (H, W, C)
            fig_img = Image.fromarray(fig_arr.astype(np.uint8))
            fig_img = fig_img.resize((width, height))
            # Create composite image with figure and env's render img
            composite_img = Image.new("RGB", (width*2, height))
            composite_img.paste(fig_img, (0,0))
            render_img = Image.fromarray(imgs[idx, :].astype(np.uint8))
            render_img = render_img.resize((width, height))
            composite_img.paste(render_img, (width, 0))
            composite_imgs.append(np.array(composite_img))
            
        composite_imgs = np.stack(composite_imgs) # (Ep*N, H, W, C)
        composite_imgs = rearrange(composite_imgs, 'B h w c -> B c h w')
        logger.log_video('eval/cluster_labels', composite_imgs, step)
        self.log_rew_grads(agent, obs, logger, step)

    def log_rew_grads(self, agent, obs, logger, step):
        num_batches = obs.shape[0]
        # Get sub-reward predictions
        obs.requires_grad = True
        sub_rews = []
        sub_rew_grads = []
        for idr in range(agent.reward_decoder_num_rews):
            features = agent.critic.encoder(obs) # (Ep*N, d)
            # Get cluster labels
            cluster_labels = torch.cdist(features, agent.reward_decoder_centroids, p=2)
            cluster_labels = torch.exp(-cluster_labels**2) # (Ep*N, d)
            obs.grad = None
            # pred_rew = agent.reward_decoder[idr](cluster_labels[:,idr].view(-1,1))
            # TODO: Visualizing cluster labels here
            pred_rew = cluster_labels[:,idr].view(-1,1)
            pred_rew.sum(dim=0).backward(retain_graph=False)
            sub_rew_grads.append(obs.grad)
            sub_rews.append(pred_rew)

        sub_rews = torch.stack(sub_rews).permute((1,0,2)).squeeze(-1) # (EP*N, num_rews)
        sub_rew_grads = torch.stack(sub_rew_grads) # (r, ep*n, c, h, w)
        num_frames = self.env.__dict__['_k']
        # Stack frames along width
        sub_rew_grads = rearrange(sub_rew_grads, 'r B (f c) h w -> r B c h (f w)', f=num_frames) # (ep*n, r, c, h, w)


        # Plot the gradients
        # Normalize gradients. Then map to [0,255]
        sub_rew_grads = sub_rew_grads.detach().cpu().abs()
        sub_rew_grads = sub_rew_grads.sum(axis=2, keepdim=True).numpy() # Sum over the channels
        normalizers = sub_rew_grads.max(axis=(1,2,3,4)) # Max over all but reward dim
        # normalizers = np.clip(normalizers, a_min=1e-8, a_max=None)# Prevent division by 0
        sub_rew_grads = sub_rew_grads/ normalizers[:,None,None,None,None]
        num_rews = sub_rew_grads.shape[0]
        sub_rew_grads = rearrange(sub_rew_grads, 'r B c h w -> B (r h) w c') # Stack sub-rewards along the 
        sub_rew_grads = sub_rew_grads*255.0
        sub_rew_grads = sub_rew_grads.repeat(3, axis=-1) # Repeat for RGB channels


        # Blend the heatmap over the render image
        bgd_imgs = obs.detach().cpu().numpy() # (B, (f c), h , w)
        bgd_imgs = bgd_imgs[None,:,:,:,:].repeat(num_rews, axis=0) # Add a sub-reward dim
        bgd_imgs = rearrange(bgd_imgs, 'r B (f c) h w -> B (r h) (f w) c', f=num_frames) # stack frames -> width, sub-rews -> height
        
        bgd_imgs = rearrange(bgd_imgs, 'B h w c -> (B h) w c')
        sub_rew_grads = rearrange(sub_rew_grads, 'B h w c -> (B h) w c')
        sub_rew_grads = Image.fromarray(sub_rew_grads.astype(np.uint8), mode='RGB')
        bgd_imgs = Image.fromarray(bgd_imgs.astype(np.uint8), mode='RGB')

        plot_imgs = Image.blend(bgd_imgs, sub_rew_grads, 0.6)
        plot_imgs = rearrange(np.array(plot_imgs), '(B h) w c -> B c h w', B=num_batches)
        
        logger.log_video(f'eval/cluster_label_grads', plot_imgs, step)


    def collect_rollout(self, agent, num_eval_episodes=1, max_episode_steps=100):
        all_imgs, all_obs, all_rews = [], [], []
        for ide in range(num_eval_episodes):
            obs = self.env.reset()
            ep_imgs, ep_obs, ep_rews = [], [], []
            for i in range(max_episode_steps):
                ep_imgs.append(self.env.render())
                ep_obs.append(obs)

                action = agent.select_action(obs)
                obs, reward, terminated, truncated, info = self.env.step(action)
                ep_rews.append(reward)
                if terminated or truncated:
                    break
            all_obs.append(np.stack(ep_obs))
            all_rews.append(np.stack(ep_rews))
            all_imgs.append(np.stack(ep_imgs))
        all_obs = np.stack(all_obs) # (Ep, N, H, W, C)
        all_rews = np.stack(all_rews) # (Ep, N, H, W, C)
        all_imgs = np.stack(all_imgs) # (Ep, N, H, W, C)

        return {'obs': all_obs, 'imgs': all_imgs, 'rews': all_rews}