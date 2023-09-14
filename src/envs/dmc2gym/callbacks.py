import torch 
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from encoder import _CLUSTER_ENCODERS
import pickle
import os

cluster_encoder_names = [x.__name__ for x in _CLUSTER_ENCODERS.values()]

class DMCCallback():

    def __init__(self, config=None):
        self.env = None
        config = config or {}
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'eval_actions.pkl'), 'rb') as f:
            self.eval_actions = pickle.load(f)
        self.eval_actions = self.eval_actions[:100]


    def set_env(self, env):
        self.env = env
        # Save a fixed sequence of actions for evaluation:
        # eval_actions = [self.env.action_space.sample() for _ in range(100)]
        # with open('eval_actions.pkl', 'wb') as f:
        #     pickle.dump(eval_actions, f)

    def __call__(self, agent, logger, step):
        assert self.env is not None, "Environment not set!"

        # obs, _ = self.env.reset()
        # done = False
        # all_obs = []
        # while not done:
        #     action = agent.act(obs)
        #     obs, _, done, _ = self.env.step(action)
        #     all_obs.append(obs)

        self.log_artifacts(agent, logger, step)

    def log_artifacts(self, agent, logger, step):
        actions = self.eval_actions
        # Log features
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

