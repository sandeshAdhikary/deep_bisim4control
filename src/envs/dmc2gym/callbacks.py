import torch 
import numpy as np
from torch.utils.data import DataLoader
from einops import rearrange
from PIL import Image
from src.models.encoder import _CLUSTER_ENCODERS
import pickle
import os

cluster_encoder_names = [x.__name__ for x in _CLUSTER_ENCODERS.values()]

class DMCCallback():

    def __init__(self, config=None):
        self.env = None
        config = config or {}
        self.domain_name = config.get('domain_name')
        self.task_name = config.get('task_name')

        assert not any([self.domain_name is None, self.task_name is None]), "Must specify domain_name and task_name in config"

        self.eval_actions_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'eval_actions_{self.domain_name}_{self.task_name}.pkl')
        self.set_eval_actions()
        
    def set_eval_actions(self):
        try:
            with open(self.eval_actions_file, 'rb') as f:
                self.eval_actions = pickle.load(f)
            self.eval_actions = self.eval_actions
        except FileNotFoundError:
            self.eval_actions = None

    def set_env(self, env):
        self.env = env
        self.is_vec_env = hasattr(self.env, 'envs')

        if self.eval_actions is None:
            # Save a fixed sequence of actions for evaluation:
            eval_actions = [self.env.action_space.sample() for _ in range(300)]
            with open(self.eval_actions_file, 'wb') as f:
                pickle.dump(eval_actions, f)
            self.set_eval_actions()

    def __call__(self, agent, logger, step):
        assert self.env is not None, "Environment not set!"
        self.log_artifacts(agent, logger, step)

    def log_artifacts(self, agent, logger, step):
        actions = self.eval_actions
        assert actions is not None, "Eval actions not set!"
        all_obs = self.collect_rollouts(agent, actions)

        embeddings = self.get_embeddings(all_obs, agent, get_feature_grads=True)

        # Log gradients
        logger.log_table('eval/features', embeddings['features'], step)

        obs_grads = embeddings['obs_grads']
        obs_grads = abs(obs_grads).sum(dim=1).mean(dim=0) # Add gradients across channels, mean over batches
        if len(obs_grads) > 0:
            logger.log_table('eval/obs_grads', obs_grads, step)

    def collect_rollouts(self, agent, actions_input):
        
        # Vectorize actions if needed
        actions = np.array(actions_input)
        if self.is_vec_env:    
            if actions.shape[1:] != self.env.action_space.shape:
                # Repeat the actions for each environment
                actions = actions[:,None,:].repeat(self.env.num_envs, axis=1)
                assert actions.shape[1:] == self.env.action_space.shape
        
        obs, info = self.env.reset()
        all_obs = [torch.from_numpy(obs).to(agent.device).to(torch.float32)]
        for ida, a in enumerate(actions):
            obs, _, terminated, truncated, _ = self.env.step(a)
            all_obs.append(torch.from_numpy(obs).to(agent.device).to(torch.float32))
            if (not self.is_vec_env) and (terminated or truncated):
                # If vec_env, then the envs are reset automatically
                obs, info = self.env.reset()
        all_obs = torch.stack(all_obs)
        return all_obs

    def get_embeddings(self, obs_input, agent, get_feature_grads=False):

        obs = obs_input
        if self.is_vec_env:
            # Stack the separate env-observations as batches
            obs = rearrange(obs, 'b n c h w -> (b n) c h w')


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

