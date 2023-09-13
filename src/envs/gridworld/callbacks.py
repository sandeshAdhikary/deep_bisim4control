import torch 
import numpy as np
from envs.gridworld.gridworld import GridWorld, GridWorldRGB
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from encoder import _CLUSTER_ENCODERS

cluster_encoder_names = [x.__name__ for x in _CLUSTER_ENCODERS.values()]

class GridWorldEvalCallback():

    def __init__(self, config=None):
        self.env = None
        config = config or {}
        self.max_features = config.get('max_features', 10) # Only display first n embeddings

        self.eval_actions = [0.06800008565187454, 0.5161293745040894, -0.04074721038341522, 0.45184072852134705, 
                             -0.29688841104507446, -0.5872771143913269, -0.27066999673843384, 0.9743776917457581,
                             0.6595609784126282, -0.5090059638023376, -0.3084663152694702, 0.9431703686714172, 
                             -0.15120454132556915, 0.14979501068592072, -0.9392542839050293, 0.15859505534172058, 
                             -0.4329833686351776, -0.8641695976257324, 0.8098902702331543, 0.7913541197776794,
                             0.651616632938385, -0.5278995633125305, -0.9936147928237915, 0.15726394951343536, 
                             0.510723888874054, 0.8965561389923096, 0.4373166859149933, -0.34975191950798035, 
                             -0.5155066847801208, 0.9070150852203369, -0.8964914083480835, -0.6602553129196167,
                             -0.057213373482227325, 0.46164175868034363, -0.6633381843566895, -0.6313488483428955, 
                             -0.8713574409484863, 0.16490218043327332, 0.7785431742668152, -0.27197033166885376,
                             -0.5168615579605103, 0.9505382776260376, -0.09180743247270584, -0.30431997776031494,
                             0.9110639691352844, 0.8347934484481812, -0.8085757493972778, 0.8504067063331604,
                             -0.69174724817276, -0.07378965616226196, -0.28244635462760925, -0.6941518783569336,
                             -0.42613720893859863, 0.7154520153999329, -0.7231209874153137, -0.9653246402740479,
                             0.7566764950752258, 0.4851411283016205, 0.8150560855865479, -0.11090514063835144,
                             -0.3010883033275604, 0.636664628982544, -0.22387298941612244, -0.688126266002655,
                             0.9242575764656067, -0.9047414660453796, 0.07533647865056992, 0.33284562826156616, 
                             -0.9387264251708984, -0.47193679213523865, -0.07526624947786331, 0.0637420192360878,
                             0.9796175956726074, 0.6180892586708069, -0.07383287698030472, -0.5457937717437744,
                             -0.6919598579406738, 0.3738366365432739, -0.4948025047779083, -0.5962380170822144,
                             -0.788661777973175, -0.5981016755104065, -0.562017023563385, 0.0012188475811854005,
                             -0.7704671025276184, 0.23730577528476715, -0.5626838207244873, -0.5957273244857788,
                             0.8529616594314575, 0.06810876727104187, 0.6587470769882202, 0.9332891702651978, 
                             -0.6421435475349426, -0.9856705665588379, 0.19447726011276245, 0.40425771474838257, 
                             -0.05647607520222664, -0.7361782789230347, 0.5463316440582275, 0.927898108959198]
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
        
        features = embeddings['features']

        # Get heatmap for all embeddings
        heatmap = self.get_heatmaps(all_poses, features, color_mode='continuous')
        # heatmap = torch.from_numpy(heatmap)
        logger.log_image('eval/embeddings', heatmap, step, image_mode='chw')

        self.log_artifacts(agent, logger, step)

        # if agent.reward_decoder_num_rews > 1:

        #     #TODO: Clustering method
        #     # Get heatmap for all predicted rewards
        #     # cluster_labels = embeddings['cluster_labels']
        #     # if cluster_labels is not None:
        #     #     heatmap = self.get_heatmaps(all_poses, cluster_labels, color_mode='continuous')
        #     #     heatmap = torch.from_numpy(heatmap)
        #     #     logger.log_image('eval/cluster_labels', heatmap, step, image_mode='chw')

            
        #     # Get heatmap for all predicted rewards
        #     pred_rews = embeddings['pred_rews']
        #     if pred_rews is not None:
        #         heatmap = self.get_heatmaps(all_poses, pred_rews, color_mode='continuous')
        #         heatmap = torch.from_numpy(heatmap)
        #         logger.log_image('eval/pred_rews', heatmap, step, image_mode='chw')

        #         # TODO: Clustering method
        #         # # Get reward grads
        #         # self.log_rew_grads(agent, all_obs.to(torch.float32), logger, step)

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

            if ido == 0:
                max_features = min(self.max_features, features.shape[1])
            all_features.append(features[:,:max_features].detach().cpu())        
        all_features = torch.concat(all_features, dim=0)

        if get_feature_grads:
            obs_grads = torch.concat(obs_grads, dim=0)


        # all_features = rearrange(all_features, 'n b d -> (n b) d')

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