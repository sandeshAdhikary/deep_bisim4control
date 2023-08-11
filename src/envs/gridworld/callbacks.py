import torch 
import numpy as np
from envs.gridworld.gridworld import GridWorld, GridWorldRGB
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
class GridWorldEvalCallback():

    def __init__(self, config=None):
        self.env = None
        config = config or {}
        self.max_embeddings = config.get('max_embeddings', 5) # Only display first n embeddings

    def set_env(self, env):
        self.env = env

    def __call__(self, agent, logger, step):
        assert self.env is not None, "Environment not set!"

        # # Turn off random init for the base environment
        # env = self.env
        # while not isinstance(env, (GridWorld,GridWorldRGB)):
        #     env = env.env
        # orig_random_init = env.random_init
        # env.random_init = False
        orig_random_init = self.env.env.env.env.env.random_init
        self.env.env.env.env.env.random_init = False

        # Loop through all grid positions
        self.env.reset()
        all_pos = list(self.env.state_iterator)
        all_obs = []
        # all_imgs = []
        all_poses = []
        for pos in all_pos:
            self.env.env.env.env.env.set_default_init_pos(pos)
            obs = self.env.reset()
            obs = torch.from_numpy(obs).to(agent.device)
            all_obs.append(obs)
            # all_imgs.append(self.env.render())
            all_poses.append(self.env.pos)
        all_obs = torch.stack(all_obs) # shape (B,3,H,W)
        # all_imgs = np.stack(all_imgs) # shape (B,H,W,3)

        # Get embeddings for all observations
        batch_size = 100
        dataloader = DataLoader(all_obs, batch_size=batch_size, shuffle=False)
        all_embeddings, all_pred_rews, all_cluster_labels = [], [], []
        for ido, obs in enumerate(dataloader):
            features = agent.critic.encoder(obs.to(agent.device))
            # TODO: decode_rewards should only take single features
            pred_rews = agent.decode_reward(features, next_features=features, sum=False)
            cluster_labels = torch.cdist(features, agent.reward_decoder_centroids, p=2)
            cluster_labels = torch.exp(-cluster_labels**2)
            

            if ido == 0:
                max_embeddings = min(self.max_embeddings, features.shape[1])
            all_embeddings.append(features[:,:max_embeddings].detach().cpu())
            all_pred_rews.append(pred_rews.detach().cpu())
            all_cluster_labels.append(cluster_labels.detach().cpu())
        all_embeddings = torch.stack(all_embeddings)
        all_embeddings = rearrange(all_embeddings, 'n b d -> (n b) d')
        all_pred_rews = torch.stack(all_pred_rews)
        all_pred_rews = rearrange(all_pred_rews, 'n b d -> (n b) d')
        all_cluster_labels = torch.stack(all_cluster_labels)
        all_cluster_labels = rearrange(all_cluster_labels, 'n b d -> (n b) d')
        if all_pred_rews.shape[-1] > 1:
            # Also append the total rewards for visualization
            all_pred_rews = torch.concat([all_pred_rews, all_pred_rews.sum(-1, keepdim=True)], dim=1)

        
        # Get heatmap for all embeddings
        heatmap = self.get_heatmaps(all_poses, all_embeddings, color_mode='continuous')
        heatmap = torch.from_numpy(heatmap)
        logger.log_image('eval/embeddings', heatmap, step)

        # Get heatmap for all predicted rewards
        heatmap = self.get_heatmaps(all_poses, all_pred_rews, color_mode='continuous')
        heatmap = torch.from_numpy(heatmap)
        logger.log_image('eval/pred_rews', heatmap, step)


        # Get heatmap for all predicted rewards
        heatmap = self.get_heatmaps(all_poses, all_cluster_labels, color_mode='continuous')
        heatmap = torch.from_numpy(heatmap)
        logger.log_image('eval/cluster_labels', heatmap, step)

        self.env.env.env.env.env.random_init = orig_random_init

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