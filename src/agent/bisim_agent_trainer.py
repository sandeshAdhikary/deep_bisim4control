# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from glob import glob

from src.utils import utils
from src.models.sac_ae import  Actor, Critic, LOG_FREQ
from src.models.transition_model import make_transition_model
from sklearn.cluster import MiniBatchKMeans, KMeans
import os
from contextlib import nullcontext

class eval_mode_context(object):
    """
    """
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(
            tau * param.data + (1 - tau) * target_param.data
        )


class BisimAgent(object):
    """Bisimulation metric algorithm."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        transition_model_type,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        encoder_stride=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_latent_lambda=0.0,
        decoder_weight_lambda=0.0,
        num_layers=4,
        num_filters=32,
        bisim_coef=0.5,
        encoder_mode='spectral',
        encoder_kernel_bandwidth='auto',
        encoder_normalize_loss=True,
        encoder_ortho_loss_reg=1e-3,
        reward_decoder_num_rews=1,
        encoder_output_dim=None,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_latent_lambda = decoder_latent_lambda
        self.transition_model_type = transition_model_type
        self.bisim_coef = bisim_coef

        self.encoder_mode = encoder_mode
        self.encoder_kernel_bandwidth = encoder_kernel_bandwidth
        self.encoder_normalize_loss = encoder_normalize_loss
        self.encoder_ortho_loss_reg = encoder_ortho_loss_reg
        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride, encoder_output_dim=encoder_output_dim
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, encoder_output_dim=encoder_output_dim
        ).to(device)

        self.reward_decoder_num_rews = reward_decoder_num_rews
        assert self.reward_decoder_num_rews > 0
        self.reward_decoder_centroids = None

        if self.reward_decoder_num_rews == 1:
            # Only network outputting single reward
            self.reward_decoder = nn.Sequential(
                nn.Linear(encoder_output_dim, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, 1)).to(device)
        else:
            self.reward_decoder = nn.ModuleList(
                [nn.Sequential(
                    nn.Linear(1, 8),
                    nn.LayerNorm(8),
                    nn.ReLU(),
                    nn.Linear(8, 1),
                    ).to(device) for _ in range(encoder_output_dim)
                ]
            )
            # self.reward_decoder_clusterer = MiniBatchKMeans(n_clusters=self.reward_decoder_num_rews, 
            #                                                 init="k-means++", 
            #                                                 n_init="auto",
            #                                                 batch_size=512,
            #                                                 random_state=123)


        # Load agent if agent_load_path specified        
        # Set up target networks for actor and critic
        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, encoder_output_dim=encoder_output_dim
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_output_dim, action_shape
        ).to(device)



        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # optimizer for critic encoder for reconstruction loss
        self.encoder_optimizer = torch.optim.Adam(
            self.critic.encoder.parameters(), lr=encoder_lr
        )

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)

    def eval(self):
        # TODO: The with eval_context sets state back to original
        #       Do we need this? Since we set .train() during start of every step?
        self.training = False
        self.actor.eval()
        self.critic.eval()

    def decode_reward(self, features, next_features=None, sum=True):
        if self.reward_decoder_num_rews == 1:
            return self.reward_decoder(features).view(-1,1)
        else:
            assert next_features is not None
            

            if hasattr(self.critic.encoder, "clusterer"):
                # # If encoder has a 'clusterer', the output is already soft-cluster labels
                rew_features = features
            else:
                # Update centroids if they don't exist
                if self.reward_decoder_centroids is None:
                    self._update_reward_decoder_centroids(next_features)

                # Features for the rewarder are distances to centroids
                rew_features = torch.cdist(next_features, self.reward_decoder_centroids, p=2)
                rew_features = torch.exp(-rew_features**2)



            # Get individual sub-rewards
            rew = []
            for idr in range(self.reward_decoder_num_rews):
                rew.append(self.reward_decoder[idr](rew_features[:,idr].view(-1,1)))
            rew = torch.stack(rew).permute((1,0,2))
            rew = rew.sum(dim=1) if sum else rew.squeeze(-1)# (B,N)
            return rew

    def _update_reward_decoder_centroids(self, features, reset=False):
        if hasattr(self.critic.encoder, "clusterer"):
            # The encoder already does clustering, so no need to keep centroids for rewarder's clustrer
            return None
        with torch.no_grad():
            if reset:
                # Reset the clusterer
                init = "k-means++" if self.reward_decoder_centroids is None else self.reward_decoder_centroids.detach().cpu().numpy()
                self.reward_decoder_clusterer = MiniBatchKMeans(n_clusters=self.reward_decoder_num_rews, 
                                                                init=init, 
                                                                n_init="auto",
                                                                batch_size=features.size(0),
                                                                random_state=123)
            
            # Update centroids
            self.reward_decoder_clusterer.partial_fit(features.detach().cpu().numpy())
            self.reward_decoder_centroids = torch.from_numpy(self.reward_decoder_clusterer.cluster_centers_).to(features.device)



    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs, batched=False):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0) if not batched else obs # Add batch dimension if unbatched
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            mu = mu.cpu().data.numpy()
            if not batched:
                mu = mu.flatten()
            return mu

    def sample_action(self, obs, batched=False, eval_mode=True):
        # TODO: is eval_mode_context() needed?
        eval_context = eval_mode_context(self) if eval_mode else nullcontext()
        with eval_context:
            with torch.no_grad():
                    obs = torch.FloatTensor(obs).to(self.device)
                    obs = obs.unsqueeze(0) if not batched else obs # Add batch dimension if unbatched
                    mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
                    pi = pi.cpu().data.numpy()
                    if not batched:
                        pi = pi.flatten() # remove batch dimension
                    return pi

    def update_critic(self, obs, action, reward, next_obs, not_done, step):
        update_info = {}

        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        # L.log('train_critic/loss', critic_loss, step)
        update_info['loss'] = critic_loss.item()

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # self.critic.log(L, step)

        return update_info

    def update_actor_and_alpha(self, obs, step):
        update_info = {}

        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()
        update_info['loss'] = actor_loss
        update_info['target_entropy'] = self.target_entropy
        # L.log('train_actor/loss', actor_loss, step)
        # L.log('train_actor/target_entropy', self.target_entropy, step)

        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        update_info['entropy'] = entropy.mean().item()
        # L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        update_info['alpha_loss'] = alpha_loss.item()
        update_info['alpha_value'] = self.alpha.item()
        # L.log('train_alpha/loss', alpha_loss, step)
        # L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()
        return update_info


    def update_encoder(self, obs, action, reward, step):
        if self.encoder_mode == 'dbc':
            return self.update_encoder_dbc(obs, action, reward, step)
        elif self.encoder_mode == 'spectral':
            return self.update_encoder_spectral(obs, action, reward, step)
        else:
            raise ValueError(f'Invalid encoder mode {self.encoder_mode}')

    def update_encoder_dbc(self, obs, action, reward, step):
        update_info = {}

        h = self.critic.encoder(obs)            

        # Sample random states across episodes at random
        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]

        with torch.no_grad():
            # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            # reward = self.decode_rewards(pred_next_latent_mu1)
            reward2 = reward[perm]
        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        z_dist = F.smooth_l1_loss(h, h2, reduction='none')
        r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')
        if self.transition_model_type == '':
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        else:
            transition_dist = torch.sqrt(
                (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
            )
            # transition_dist  = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none') \
                # +  F.smooth_l1_loss(pred_next_latent_sigma1, pred_next_latent_sigma2, reduction='none')

        bisimilarity = r_dist + self.discount * transition_dist
        loss = (z_dist - bisimilarity).pow(2).mean()
        update_info['encoder_loss'] = loss.item()
        # L.log('train_ae/encoder_loss', loss, step)
        return loss, update_info


    def update_encoder_spectral(self, obs, action, reward, step):
        update_info = {}

        h = self.critic.encoder(obs)        

        # Sample random states across episodes at random
        batch_size = obs.size(0)

        with torch.no_grad():
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
        
        with torch.no_grad():

            # Get bi-sim distances
            r_dist = torch.cdist(reward, reward, p=1) # shape (B,B)
            if self.transition_model_type in ['', 'deterministic']:
                transition_dist = torch.cdist(pred_next_latent_mu1, pred_next_latent_mu1, p=2) # shape (B,B)
            else:
                mu_dist = torch.cdist(pred_next_latent_mu1, pred_next_latent_mu1, p=2)
                sigma_sqrt = torch.sqrt(pred_next_latent_sigma1)
                sigma_dist = torch.cdist(sigma_sqrt, sigma_sqrt, p=2)
                transition_dist = torch.sqrt(mu_dist**2 + sigma_dist**2)
                
            bisimilarity = r_dist + self.discount*transition_dist # shape (B,B)

        
            # Get kernel/weights matrix
            kernel_bandwidth = self.encoder_kernel_bandwidth
            if kernel_bandwidth == 'auto':
                kernel_bandwidth = (2*(torch.median(bisimilarity)*2))
                kernel_bandwidth = kernel_bandwidth if kernel_bandwidth > 0 else 1.0
                nu = 1./kernel_bandwidth
            else:
                nu = 1./(2*(kernel_bandwidth**2))
            # L.log('train_ae/kernel_nu', nu, step)
            W = torch.exp(-nu*(bisimilarity)**2) # shape (B, B)

        if self.encoder_normalize_loss:
            D = torch.sum(W, dim=1)
            h = h / D[:, None]

        Dh = torch.cdist(h, h, p=2)
        loss = torch.sum(W * Dh.pow(2)) / (batch_size**2)
        update_info['dist_loss'] = loss.item()
        # L.log('train_ae/dist_loss', loss.item(), step)

        # Add orthogonality loss
        if self.encoder_ortho_loss_reg > 0:    
            D = torch.diag(torch.sum(W, dim=1))
            est = (h.T @ D @ h)
            ortho_loss = (est - torch.eye(h.shape[1]).to(h.device))**2
            # ortho_loss = (((Y.T@Y)*(1/Y.shape[0]) - th.eye(Y.shape[1]).to(Y.device))**2)
            ortho_loss = ortho_loss.sum()/(h.shape[1])
            loss = loss + self.encoder_ortho_loss_reg*ortho_loss
            update_info['ortho_loss'] = ortho_loss.item()
            # L.log('train_ae/ortho_loss', ortho_loss, step)
        update_info['encoder_loss'] = loss.item()
        # L.log('train_ae/encoder_loss', loss, step)
        
        return loss, update_info

    def update_transition_reward_model(self, obs, action, next_obs, reward, step):
        update_info = {}

        h = self.critic.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        update_info['transition_loss'] = loss.item()
        # L.log('train_ae/transition_loss', loss, step)

        # Predict next reward from next latent
        # pred_next_latent = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        # pred_next_reward = self.decode_reward(pred_next_latent)
        # reward_loss = F.mse_loss(pred_next_reward, reward)
        # loss = loss + reward_loss
        # L.log('train_ae/reward_loss', reward_loss, step)    

        # Predict reward directly from latent and action
        pred_reward = self.decode_reward(h, next_features=next_h)
        # pred_reward = self.decode_reward(h.detach(), next_features=next_h.detach())
        reward_loss = F.mse_loss(pred_reward, reward)
        loss = loss + reward_loss
        update_info['reward_loss'] = reward_loss.item()
        # L.log('train_ae/reward_loss', reward_loss, step)

        return loss, update_info

    def update(self, batch, step):
        """
        Note: The old version took replay buffer as input; we supply sampled batch
        """

        update_step_info = {}

        obs, action, _, reward, next_obs, not_done = batch
        update_step_info['batch_reward'] = reward.mean().item()
        
        critic_update_info = self.update_critic(obs, action, reward, next_obs, not_done, step)
        transition_reward_loss, transition_update_info = self.update_transition_reward_model(obs, action, next_obs, reward,step)
        encoder_loss, encoder_update_info = self.update_encoder(obs, action, reward, step)
        total_loss = self.bisim_coef * encoder_loss + transition_reward_loss
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        actor_and_alpha_info = None
        if step % self.actor_update_freq == 0:
            actor_and_alpha_info = self.update_actor_and_alpha(obs, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        update_step_info = {
            'critic': critic_update_info,
            'transition': transition_update_info,
            'encoder': encoder_update_info,
            'actor_and_alpha': actor_and_alpha_info,
        }

        return update_step_info
            
    def save(self, model_dir, filename=None, step=None, save_optimizers=False):
        filename = filename or f'model_{step}.pt'
        torch.save(self.state_dict(save_optimizers=save_optimizers),
                    f'{model_dir}/{filename}')
                
    def state_dict(self, save_optimizers=True):
        constructor_params = {'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict(),
                            'critic_target': self.critic_target.state_dict(),
                            'reward_decoder': self.reward_decoder.state_dict(),
                            'transition_model': self.transition_model.state_dict(),
                            'log_alpha': self.log_alpha,
                            'target_entropy': self.target_entropy,
                            }

        if save_optimizers:
            constructor_params.update({
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
                'decoder_optimizer': self.decoder_optimizer.state_dict(),
                'encoder_optimizer': self.encoder_optimizer.state_dict()
                })

        model_dict = {name: value for (name,value) in constructor_params.items()}
        return model_dict


    def load(self, state_dict=None, model_file= None, model_dir=None, step=None, checkpoint=False):

        if checkpoint:
            step = 'chkpt'
 
        if state_dict is not None:
            model_dict = state_dict
        elif model_file is not None:
            model_dict = torch.load(model_file)    
        else:
            # # Load the model_dict
            model_dict = torch.load(f'{model_dir}/model_{step}.pt')

        for name,value in model_dict.items():
            module = getattr(self, name)
            # assert not torch.isnan(value).any(), f"NaNs in {name}"
            if hasattr(module, 'load_state_dict'):
                # any_nans = any([torch.isnan(x).any().item() for x in value.values()])
                # assert not any_nans, f"NaNs in {name}"
                module.load_state_dict(value)
                # assert not any([v.isnan().any() for v in module.state_dict().values()])      
            elif isinstance(module, (torch.Tensor,)):
                device = getattr(module, 'device') 
                dtype = getattr(module, 'dtype')
                if hasattr(module, 'data'):
                    setattr(module, 'data', value)
                else:
                    setattr(self, name, torch.tensor(value, device=device, dtype=dtype))
            else:
                setattr(self, name, value)    
            # else:
            #     raise ValueError(f"Unknown module type for checkpoint loading: {name}")
            
        # print("Done")

    ###########
    def save_checkpoint(self, model_dir):
        self.save(model_dir, checkpoint=True)

    def load_checkpoint(self, model_dir):
        self.load(model_dir, checkpoint=True)