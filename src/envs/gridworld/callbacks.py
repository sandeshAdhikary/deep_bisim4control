import torch 
import numpy as np
from envs.gridworld.gridworld import GridWorld, GridWorldRGB
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from encoder import _CLUSTER_ENCODERS
import pickle
import os

cluster_encoder_names = [x.__name__ for x in _CLUSTER_ENCODERS.values()]

class GridWorldEvalCallback():

    def __init__(self, config=None):
        self.env = None
        config = config or {}
        self.max_features_display = config.get('max_features', 10) # Only display first n embeddings

        # eval_actions = [self.env.action_space.sample() for _ in range(100)]
        # with open('eval_actions.pkl', 'wb') as f:
        #     pickle.dump(eval_actions, f)

        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_actions.pkl'), 'rb') as f:
            self.eval_actions = pickle.load(f)
        self.eval_actions = self.eval_actions[:30]

    def set_env(self, env):
        self.env = env


    def __call__(self, agent, logger, step):
        assert self.env is not None, "Environment not set!"

        # Temporarily disable random initialization
        orig_random_init = self.env.env.env.env.env.random_init
        self.env.env.env.env.env.random_init = False

        # Loop through all grid positions
        self.env.reset()
        all_pos = list(self.env.state_iterator)
        all_obs = []
        all_poses = []
        for pos in all_pos:
            self.env.env.env.env.env.set_default_init_pos(pos)
            obs = self.env.reset()
            obs = torch.from_numpy(obs).to(agent.device)
            all_obs.append(obs)
            all_poses.append(self.env.pos)
        all_obs = torch.stack(all_obs) # shape (B,3,H,W)

        embeddings = self.get_embeddings(all_obs, agent)
        
        features = embeddings['features'][:self.max_features_display] # Only display first n features

        # Get heatmap for all embeddings
        heatmap = self.get_heatmaps(all_poses, features, color_mode='continuous')
        logger.log_image('eval/embeddings', heatmap, step, image_mode='chw')

        self.log_artifacts(agent, logger, step)

        self.env.env.env.env.env.random_init = orig_random_init

    def log_artifacts(self, agent, logger, step):
        actions = self.eval_actions
        # Log features
        self.env.env.env.env.env.set_default_init_pos([0.,0.])
        obs = self.env.reset()
        all_obs = [torch.from_numpy(obs).to(agent.device).to(torch.float32)]
        for ida, a in enumerate(actions):
            obs, _, terminated, truncated, _ = self.env.step(a)
            all_obs.append(torch.from_numpy(obs).to(agent.device).to(torch.float32))
            if terminated or truncated:
                self.env.reset()
        all_obs = torch.stack(all_obs)

        embeddings = self.get_embeddings(all_obs, agent, get_feature_grads=True)

        # Log gradients
        logger.log_table('eval/features', embeddings['features'], step)

        obs_grads = embeddings['obs_grads']
        obs_grads = abs(obs_grads).sum(dim=1).mean(dim=0) # Add gradients across channels, mean over batches
        if len(obs_grads) > 0:
            logger.log_table('eval/obs_grads', obs_grads, step)

    def get_embeddings(self, obs, agent, get_feature_grads=False):

        # Get embeddings for all observations
        batch_size = 100
        dataloader = DataLoader(obs, batch_size=batch_size, shuffle=False)
        all_features = []
        if agent.reward_decoder_num_rews > 1:
            all_pred_rews, all_cluster_labels = [], []
        else:
            all_pred_rews, all_cluster_labels = None, None

        if get_feature_grads:
            obs.requires_grad = True

        obs_grads = []
        for ido, obs in enumerate(dataloader):
            
            agent.critic.zero_grad()
            # features = agent.critic.encoder(obs.to(agent.device))
            features = agent.critic.encoder(obs)

            if get_feature_grads:
                obs.retain_grad()
                features.sum().backward()
                obs_grads.append(obs.grad.detach().cpu())
            all_features.append(features.detach().cpu())        
        all_features = torch.concat(all_features, dim=0)
        if get_feature_grads:
            obs_grads = torch.concat(obs_grads, dim=0)
        return {'features': all_features, 'pred_rews': all_pred_rews, 'cluster_labels': all_cluster_labels, 'obs_grads': obs_grads}


    def get_heatmaps(self, indices, values, color_mode='continuous'):
        heatmaps = []
        for idc in range(values.shape[1]):
            heatmap = self._get_single_heatmap(indices, values[:,idc], color_mode=color_mode)
            heatmap = np.pad(heatmap, ((0,0),(10,10),(10,10)), mode='constant', constant_values=0)
            heatmaps.append(heatmap)
        heatmaps = np.stack(heatmaps)
        heatmaps = rearrange(heatmaps, 'n c h w -> c (n h) w')
        return heatmaps


    def _get_single_heatmap(self, indices, values, color_mode='continuous'):
        # Get heatmap
        heatmap = np.zeros((self.env.size, self.env.size))
        for s,v in zip(indices, values):
            heatmap[int(s[0]), int(s[1])] = v
        # Render heatmap
        heatmap = self.env.render_heatmap(heatmap, color_mode)
        # Overlay heatmap onto the grid
        grid = Image.fromarray(self.env.render())
        heatmap = heatmap.resize(grid.size)        
        heatmap_overlay = Image.blend(grid, heatmap, 0.5)
        # Convert to array
        heatmap_overlay = np.array(heatmap_overlay)
        heatmap_overlay = heatmap_overlay.transpose(2,0,1)
        return heatmap_overlay
    
    def log_rew_grads(self, agent, obs, logger, step):
        num_batches = obs.shape[0]
        # Get sub-reward predictions
        obs.requires_grad = True
        features = agent.critic.encoder(obs) # (Ep*N, d)
        # Get cluster labels
        if agent.critic.encoder.__class__.__name__ in cluster_encoder_names:
            # The encoder outputs are already cluster labels
            cluster_labels = features
        else:
            # Use rewarder's cluster to get cluster labels
            cluster_labels = torch.cdist(features, agent.reward_decoder_centroids, p=2)
            cluster_labels = torch.exp(-cluster_labels**2) # (Ep*N, d)
        sub_rews = []
        sub_rew_grads = []
        for idr in range(agent.reward_decoder_num_rews):
            obs.grad = None
            pred_rew = agent.reward_decoder[idr](cluster_labels[:,idr].view(-1,1))
            pred_rew.sum(dim=0).backward(retain_graph=True)
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

        plot_imgs = Image.blend(bgd_imgs, sub_rew_grads, 0.8)
        plot_imgs = rearrange(np.array(plot_imgs), '(B h) w c -> B c h w', B=num_batches)
        
        logger.log_video(f'eval/sub_rew_grads', plot_imgs, step)