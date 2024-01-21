# Original agent code from: https://github.com/facebookresearch/deep_bisim4control/
## License for original code:
## Copyright (c) Facebook, Inc. and its affiliates.
## All rights reserved.
## This source code is licensed under the license found in the
## LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.sac_ae import  Actor, Critic, ActorResidual
from src.models.transition_model import make_transition_model
import src.agent.utils as agent_utils


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
        decode_rewards_from_next_latent=True,
        residual_actor=False,
        use_schedulers=True,
        encoder_softmax=False,
        distance_type='bisim',
        predict_inverse_dynamics=False,
        inverse_dynamics_lr=1e-4,
        inverse_dynamics_loss_weight=2.0,
        encoder_max_norm=False,
        intrinsic_reward=False,
        intrinsic_reward_max=1.0,
        intrinsic_reward_scale=1.0,
        trunk_regularization=False,
        trunk_regularization_coeff=1e-3
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
        self.decode_rewards_from_next_latent = decode_rewards_from_next_latent
        self.residual_actor = residual_actor

        self.predict_inverse_dynamics = predict_inverse_dynamics
        self.inverse_dynamics_loss_weight = inverse_dynamics_loss_weight

        self.intrinsic_reward = intrinsic_reward
        self.intrinsic_reward_max = intrinsic_reward_max
        self.intrinsic_reward_scale = intrinsic_reward_scale

        self.trunk_regularization = trunk_regularization
        self.trunk_regularization_coeff = trunk_regularization_coeff

        self.encoder_max_norm = None
        if encoder_max_norm:
            # From RobustDBC (https://arxiv.org/pdf/2110.14096.pdf)
            # Assuming c_T = gamma, and R in [0,1]
            c_R = 1.0
            c_T = discount
            self.encoder_max_norm = 0.5 * c_R / (1-c_T)


        self.distance_type = distance_type
        self.mico_beta = 0.1 # beta for MICO distance

        if residual_actor:
            self.actor = ActorResidual(
                obs_shape, action_shape, hidden_dim, encoder_type,
                encoder_feature_dim, actor_log_std_min, actor_log_std_max,
                num_layers, num_filters, encoder_stride, encoder_softmax=encoder_softmax,
                encoder_max_norm=self.encoder_max_norm
            ).to(device)
        else:
            self.actor = Actor(
                obs_shape, action_shape, hidden_dim, encoder_type,
                encoder_feature_dim, actor_log_std_min, actor_log_std_max,
                num_layers, num_filters, encoder_stride, encoder_softmax=encoder_softmax,
                encoder_max_norm=self.encoder_max_norm
            ).to(device)


        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, encoder_softmax=encoder_softmax,
                encoder_max_norm=self.encoder_max_norm
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride, encoder_softmax=encoder_softmax,
                encoder_max_norm=self.encoder_max_norm
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        self.reward_decoder = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 1)).to(device)

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

        # Set up inverse dynamics network and optimizer
        if self.predict_inverse_dynamics:
            self.inverse_dynamics_predictor = self.setup_inverse_dynamics_predictor(encoder_feature_dim, 
                                                                                    action_shape)
            self.inverse_dynamics_optimizer = torch.optim.Adam( self.inverse_dynamics_predictor.parameters(),
                                                    lr=inverse_dynamics_lr, weight_decay = 1e-5 )


        # scheduler_kwargs:
        scheduler_step_size = 1000
        scheduler_gamma = 0.99

        self.actor_scheduler = torch.optim.lr_scheduler.StepLR(self.actor_optimizer, 
                                                            scheduler_step_size, gamma=scheduler_gamma) if use_schedulers else None
        self.critic_scheduler = torch.optim.lr_scheduler.StepLR(self.critic_optimizer, 
                                                                scheduler_step_size, gamma=scheduler_gamma) if use_schedulers else None
        self.log_alpha_scheduler = torch.optim.lr_scheduler.StepLR(self.log_alpha_optimizer, 
                                                                scheduler_step_size, gamma=scheduler_gamma) if use_schedulers else None
        self.decoder_scheduler = torch.optim.lr_scheduler.StepLR(self.decoder_optimizer, 
                                                            scheduler_step_size, gamma=scheduler_gamma) if use_schedulers else None
        self.encoder_scheduler = torch.optim.lr_scheduler.StepLR(self.encoder_optimizer, 
                                                                scheduler_step_size, gamma=scheduler_gamma) if use_schedulers else None
        self.inverse_dynamics_scheduler = torch.optim.lr_scheduler.StepLR(self.inverse_dynamics_optimizer, 
                                                                          scheduler_step_size, gamma=scheduler_gamma) if use_schedulers else None


        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
    
    def eval(self):
        self.training = False
        self.actor.eval()
        self.critic.eval()

    @property
    def alpha(self):
        return self.log_alpha.exp()

    # def select_action(self, obs):
    #     with torch.no_grad():
    #         obs = torch.FloatTensor(obs).to(self.device)
    #         obs = obs.unsqueeze(0)
    #         mu, _, _, _ = self.actor(
    #             obs, compute_pi=False, compute_log_pi=False
    #         )
    #         return mu.cpu().data.numpy().flatten()

    # def sample_action(self, obs):
    #     with torch.no_grad():
    #         obs = torch.FloatTensor(obs).to(self.device)
    #         obs = obs.unsqueeze(0)
    #         mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
    #         return pi.cpu().data.numpy().flatten()

    def select_action(self, obs, batched=False):
        """
        Same as original, but now allows batched inputs (e.g. from vec environments)
        """
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0) if not batched else obs # Add batch dimension if unbatched
            mu, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            mu = mu.cpu().data.numpy()
            if not batched:
                mu = mu.flatten()
            return mu

    def sample_action(self, obs, batched=False, eval_mode=True):
        """
        Same as original, but now allows batched inputs (e.g. from vec environments)
        """
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0) if not batched else obs # Add batch dimension if unbatched
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            pi = pi.cpu().data.numpy()
            if not batched:
                pi = pi.flatten() # remove batch dimension
            return pi
    



    def update_critic(self, obs, action, reward, next_obs, not_done, L=None, step=None):
        """
        Same as original; just returns output info.
        """
        output_dict = {}

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
        
        # Trunk regularization
        if (self.trunk_regularization):
            if self.critic.encoder.eigvals is not None:
                # Use eigenvalues if available
                reg_weights = self.critic.encoder.eigvals.detach()
            else:
                # else, uniform weights
                reg_weights = torch.ones(self.critic.encoder.feature_dim).to(self.device)

            bandwidth = 1./(2*torch.median(reg_weights))
            bandwidth = bandwidth if bandwidth > 0 else 1.0
            reg_weights = torch.exp(-bandwidth*reg_weights) # penalize smaller eigenvalues
            
            # Regularization for Q1
            W_1 = self.critic.Q1.trunk[0].weight # Weights of first linear layer after encoder
            W_1 = W_1[:,:self.critic.encoder.feature_dim] # Ignore the action dimensions
            trunk_reg_1 = torch.abs(W_1**2 @ reg_weights).mean()
            # Regularization for Q2
            W_2 = self.critic.Q2.trunk[0].weight # Weights of first linear layer after encoder
            W_2 = W_2[:,:self.critic.encoder.feature_dim] # Ignore the action dimensions
            trunk_reg_2 = torch.abs(W_2**2 @ reg_weights).mean()
            # Avearage the two regularization terms
            trunk_reg = (trunk_reg_1+trunk_reg_2)/2.0
            critic_loss += self.trunk_regularization_coeff*trunk_reg
            output_dict['trunk_reg'] = trunk_reg.item()

        output_dict['loss'] = critic_loss.item()
        if L is not None:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # nn.utils.clip_grad.clip_grad_norm_(self.critic.parameters(), 40.)
        self.critic_optimizer.step()
        if L is not None:
            self.critic.log(L, step)
        
        return output_dict


    def update_actor_and_alpha(self, obs, L=None, step=None):
        """
        Same as original; just returns output info instead of logging
        """
        output_dict = {}

        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        # Spectral regularization
        if self.trunk_regularization:
            if self.actor.encoder.eigvals is not None:
                # Use eigenvalues if available
                reg_weights = self.actor.encoder.eigvals.detach()
            else:
                reg_weights = torch.ones(self.actor.encoder.feature_dim).to(self.device)
            bandwidth = 1./(2*torch.median(reg_weights))
            bandwidth = bandwidth if bandwidth > 0 else 1.0
            reg_weights = torch.exp(-bandwidth*reg_weights) # penalize smaller eigenvalues
            
            W = self.actor.trunk[0].weight # Weights of first linear layer after encoder

            trunk_reg = torch.abs(W**2 @ reg_weights).mean()
            actor_loss += self.trunk_regularization_coeff*trunk_reg
            output_dict['trunk_reg'] = trunk_reg.item()

        output_dict['loss'] = actor_loss.item()
        output_dict['target_entropy'] = self.target_entropy


        if L is not None:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)


        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        output_dict['entropy'] = entropy.mean().item()
        if L is not None:
            L.log('train_actor/entropy', entropy.mean(), step)



        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        # nn.utils.clip_grad.clip_grad_norm_(self.actor.parameters(), 40.)
        self.actor_optimizer.step()
        if L is not None:
            self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        output_dict['alpha_loss'] = alpha_loss.item()
        output_dict['alpha_value'] = self.alpha.item()
        if L is not None:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        return output_dict

    def update_inverse_dynamics(self, obs, action, next_obs, L=None, step=None):
        """
        Adapted from https://github.com/metekemertas/RobustBisimulation/tree/main
        based on https://arxiv.org/pdf/2110.14096.pdf
        """
        # B, nA = action.shape
        #assert nA == 1 -> beware of action scale differences between dims
        # Get encodings
        h      = self.critic.encoder(obs) # Encoding of current observation
        next_h = self.critic.encoder(next_obs) # Encoding of next observation
        # Predict action
        pred_action = self.inverse_dynamics_predictor( torch.cat([ h, next_h ], dim = -1) )
        assert pred_action.shape == action.shape
        loss = (action - pred_action).abs().mean()
        loss_dict = {
            'inverse_dynamics_loss': loss.item()
        }
        return loss, loss_dict

    def update_encoder(self, obs, action, reward, L=None, step=None, next_obs=None):
        """
        Same as original but return output_dict in addition to the loss
        """
        output_dict = {}

        h = self.critic.encoder(obs)            

        # Sample random states across episodes at random
        batch_size = obs.size(0)
        perm = np.random.permutation(batch_size)
        h2 = h[perm]

        with torch.no_grad():
            # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            # reward = self.reward_decoder(pred_next_latent_mu1)
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

        if self.distance_type == 'bisim':
            # Loss function as used in the original DBC paper
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
                #     +  F.smooth_l1_loss(pred_next_latent_sigma1, pred_next_latent_sigma2, reduction='none')

            bisimilarity = r_dist + self.discount * transition_dist
            loss = (z_dist - bisimilarity).pow(2).mean()
        elif self.distance_type == 'mico':
            # Get the distance between the embeddings
            z_dist = 0.5*(h.pow(2).sum(dim=1) + h2.pow(2).sum(dim=1))
            z_dist += self.mico_beta * self._cosine_distance(h, h2)
            
            # Get the target MICO distance
            reward_dist = F.smooth_l1_loss(reward, reward2, reduction='none').squeeze() # The reward distance
            with torch.no_grad():
                # Note: MICO uses the frozen encoder to get these
                h_next = self.critic_target.encoder(next_obs)
                h_next_2 = h_next[perm]
            transition_dist =  0.5*(h_next.pow(2).sum(dim=1) + h_next_2.pow(2).sum(dim=1))
            transition_dist += self.mico_beta *  self._cosine_distance(h_next, h_next_2)
            mico_dist = reward_dist + self.discount * transition_dist
            
            # Get loss
            loss = (z_dist - mico_dist).pow(2).mean()
        else:
            raise ValueError(f"Unknown distance type: {self.distance_type}")

        output_dict['encoder_loss'] = loss.item()
        if L is not None:
            L.log('train_ae/encoder_loss', loss, step)

        output_dict['embedding_norm'] = torch.norm(h, dim=1).mean().item()

        return loss, output_dict

    def _cosine_distance(self, x, y, epsilon=1e-8):
        # Get the cosine distance from cosine similarities
        cos_theta = torch.cosine_similarity(x, y, dim=-1, eps=epsilon)
        tolerance = torch.ones_like(cos_theta)*epsilon # tolerance; to prevent sqrt(0)
        sin_theta = torch.sqrt(torch.maximum(1 - cos_theta.pow(2), tolerance))
        return torch.atan2(sin_theta, cos_theta) # theta = arctan(sin/cos)

    def update_transition_reward_model(self, obs, action, next_obs, reward, L=None, step=None):
        """
        Same as original except:
        1. Returns output_dict
        2. Added option to decode rewards from next_latent or current_latent
        3. Swapped self.reward_decoder() with self.decode_reward(). The latter
           is a more general function that can take next_latent state as well
        """
        output_dict = {}

        h = self.critic.encoder(obs)
        pred_next_latent_mu, pred_next_latent_sigma = self.transition_model(torch.cat([h, action], dim=1))
        if pred_next_latent_sigma is None:
            pred_next_latent_sigma = torch.ones_like(pred_next_latent_mu)

        next_h = self.critic.encoder(next_obs)
        diff = (pred_next_latent_mu - next_h.detach()) / pred_next_latent_sigma
        loss = torch.mean(0.5 * diff.pow(2) + torch.log(pred_next_latent_sigma))
        
        output_dict['transition_loss'] = loss.item()
        if L is not None:
            L.log('train_ae/transition_loss', loss, step)
        
        if self.decode_rewards_from_next_latent:
            # Predict reward from next latent
            pred_next_latent = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
            pred_reward = self.decode_reward(pred_next_latent)
        else:
            # Predict reward from current latent
            pred_reward = self.decode_reward(h)

        reward_loss = F.mse_loss(pred_reward, reward)
        output_dict['reward_loss'] = reward_loss.item()

        total_loss = loss + reward_loss
        return total_loss, output_dict
    
    def decode_reward(self, features, next_features=None, sum=True):
        return self.reward_decoder(features)

    
    def update(self, data, step, L=None, update_from_batch=True):
        """
        data is either a batch or the replay buffer
        """
        output_dict = {}
        if update_from_batch:
            obs, action, _, reward, next_obs, not_done = data
        else:
            obs, action, _, reward, next_obs, not_done = data.sample()    

        if L is not None:
            L.log('train/batch_reward', reward.mean(), step)

        output_dict = self._update(obs, action, reward, next_obs, not_done, step=step, L=L)
        return output_dict


    def _update(self, obs, action, reward, next_obs, not_done, step, L=None):
        """
        Same as original except accouting for update_dicts, and returning a composite update_dict
        """

        critic_update_dict = self.update_critic(obs, action, reward, next_obs, not_done, L, step)
        transition_reward_loss, transition_update_dict = self.update_transition_reward_model(obs, action, next_obs, reward, L, step)
        encoder_loss, encoder_update_dict = self.update_encoder(obs, action, reward, L, step, next_obs=next_obs)
        total_loss = self.bisim_coef * encoder_loss + transition_reward_loss

        if self.predict_inverse_dynamics:
            inverse_dynamics_loss, inverse_dynamics_dict = self.update_inverse_dynamics(obs, action, next_obs, L, step)
            total_loss = total_loss + self.inverse_dynamics_loss_weight * inverse_dynamics_loss
            encoder_update_dict.update(inverse_dynamics_dict)
            self.inverse_dynamics_optimizer.zero_grad()

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        total_loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        
        if self.predict_inverse_dynamics:
            self.inverse_dynamics_optimizer.step()

        actor_and_alpha_update_dict = None
        if step % self.actor_update_freq == 0:
            actor_and_alpha_update_dict = self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            agent_utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            agent_utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        # Update schedulers
        if self.critic_scheduler is not None:
            self.critic_scheduler.step()
        if self.actor_scheduler is not None:
            self.actor_scheduler.step()
        if self.log_alpha_scheduler is not None:
            self.log_alpha_scheduler.step()
        if self.decoder_scheduler is not None:
            self.decoder_scheduler.step()
        if self.encoder_scheduler is not None:
            self.encoder_scheduler.step()
        if self.inverse_dynamics_scheduler is not None:
            self.inverse_dynamics_scheduler.step()

        return {
            'critic': critic_update_dict,
            'transition': transition_update_dict,
            'encoder': encoder_update_dict,
            'actor_and_alpha': actor_and_alpha_update_dict
        }

    # def save(self, model_dir, step):
    #     """
    #     The original save function
    #     """
    #     torch.save(
    #         self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
    #     )
    #     torch.save(
    #         self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
    #     )
    #     torch.save(
    #         self.reward_decoder.state_dict(),
    #         '%s/reward_decoder_%s.pt' % (model_dir, step)
    #     )

    def save(self, model_dir, filename=None, step=None, save_optimizers=False):
        """
        Modularized state_dict() as separate function
        state_dict now also consists of all constructor_params needed to recreate model
        """
        filename = filename or f'model_{step}.pt'
        torch.save(self.state_dict(save_optimizers=save_optimizers),
                    f'{model_dir}/{filename}')

    def state_dict(self, include_optimizers=True):
        """
        state dict consists of all constructor params needed to recreate model
        """
        constructor_params = {'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict(),
                            'critic_target': self.critic_target.state_dict(),
                            'reward_decoder': self.reward_decoder.state_dict(),
                            'transition_model': self.transition_model.state_dict(),
                            'log_alpha': self.log_alpha,
                            'target_entropy': self.target_entropy,
                            }

        if include_optimizers:
            constructor_params.update({
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
                'decoder_optimizer': self.decoder_optimizer.state_dict(),
                'encoder_optimizer': self.encoder_optimizer.state_dict()
                })

        model_dict = {name: value for (name,value) in constructor_params.items()}
        return model_dict


    # def load(self, model_dir, step):
    #     """
    #     The original load function
    #     """
    #     self.actor.load_state_dict(
    #         torch.load('%s/actor_%s.pt' % (model_dir, step))
    #     )
    #     self.critic.load_state_dict(
    #         torch.load('%s/critic_%s.pt' % (model_dir, step))
    #     )
    #     self.reward_decoder.load_state_dict(
    #         torch.load('%s/reward_decoder_%s.pt' % (model_dir, step))
    #     )

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
            if hasattr(module, 'load_state_dict'):
                module.load_state_dict(value)
            elif isinstance(module, (torch.Tensor,)):
                device = getattr(module, 'device') 
                dtype = getattr(module, 'dtype')
                if hasattr(module, 'data'):
                    setattr(module, 'data', value)
                else:
                    setattr(self, name, torch.tensor(value, device=device, dtype=dtype))
            else:
                setattr(self, name, value)    


    def to(self, device):
        self.device = device
        self.actor = self.actor.to(device=device)
        self.critic = self.critic.to(device=device)
        self.critic_target = self.critic_target.to(device=device)
        self.transition_model = self.transition_model.to(device=device)
        self.reward_decoder = self.reward_decoder.to(device=device)
        self.log_alpha = self.log_alpha.to(device=device)
        return self
    

    def setup_inverse_dynamics_predictor(self, phi_dim, action_shape, hidden_layers = None):
        """
        From Kemertas et al (Neurips, 2021) https://github.com/metekemertas/RobustBisimulation/tree/main
        Kemertes et al propose inverse dynamics prediction as a way to prevent embedding collapse
        (https://arxiv.org/pdf/2110.14096.pdf)
        """

        if hidden_layers is None:
            hidden_layers = (256, 128)

        # ~Similar to Pathak et al architecture
        return nn.Sequential( # Trained 
            nn.Linear(2 * phi_dim, hidden_layers[0]),
            nn.ELU(),
            nn.Linear(hidden_layers[0], hidden_layers[1]),
            nn.ELU(),
            nn.Linear(hidden_layers[1], *action_shape)
            ).to(self.device)

    def modify_experience(self, obs, action, curr_reward, reward, next_obs, terminated, truncated, info):
        """
        Modify experience before adding to replay buffer.
        Optionally, compute intrinsic reward and add to reward.
        NOTE: No need to update curr_reward since curr_reward was already adjusted in the prior step
        """
        if self.intrinsic_reward:
            with torch.no_grad():
                # Get encodings for current observations
                h = self.critic.encoder(torch.from_numpy(obs).to(self.device))
                # Predict the next latent state
                pred_next_latent_mu, _ = self.transition_model(
                    torch.cat([h, torch.from_numpy(action).to(self.device)], dim=1)
                    )
                # Get the actual next latent
                next_h = self.critic.encoder(torch.from_numpy(next_obs).to(self.device))
                # intrinsic reward is the error in predicting the next latent
                intrinsic_reward = 0.5 * (pred_next_latent_mu - next_h).pow(2).mean(-1).detach().clamp(max=self.intrinsic_reward_max)    
                reward = reward + intrinsic_reward.cpu().numpy()*self.intrinsic_reward_scale

        return obs, action, curr_reward, reward, next_obs, terminated, truncated, info