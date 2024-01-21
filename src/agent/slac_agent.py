import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.models.sac_ae import  Actor, Critic, ActorResidual, weight_init
from src.models.transition_model import make_transition_model
import src.agent.utils as agent_utils
from src.agent.bisim_agent_baseline import BisimAgent
from src.models.decoder import make_decoder


class SLACAgent(BisimAgent):
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

        # optimizer for decoder: uses transition_model
        self.decoder_optimizer = torch.optim.Adam(
            self.transition_model.parameters(),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        self.decoder = None
        encoder_params = list(self.critic.encoder.parameters()) + list(self.transition_model.parameters())
        if decoder_type == 'pixel':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            # TODO: Add weight_init back in
            # self.decoder.apply(weight_init)
        elif decoder_type == 'inverse':
            self.inverse_model = nn.Sequential(
                nn.Linear(encoder_feature_dim * 2, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_shape[0])).to(device)
            encoder_params += list(self.inverse_model.parameters())
        if decoder_type != 'identity':
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=encoder_lr)
        if decoder_type == 'pixel':  # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
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

    def update_decoder(self, obs, action, target_obs, L, step):  #  uses transition model
        # raise NotImplementedError
        # # image might be stacked, just grab the first 3 (rgb)!
        assert target_obs.dim() == 4, target_obs.shape
        target_obs = target_obs[:, :3, :, :]

        h = self.critic.encoder(obs)
        next_h = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = agent_utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(next_h)
        loss = F.mse_loss(target_obs, rec_obs)

        # self.encoder_optimizer.zero_grad()
        # self.decoder_optimizer.zero_grad()
        # loss.backward()

        # self.encoder_optimizer.step()
        # self.decoder_optimizer.step()
        # L.log('train_ae/ae_loss', loss, step)


    def _update(self, obs, action, reward, next_obs, not_done, step, L=None):
        # Update critic
        critic_update_dict = self.update_critic(obs, action, reward, next_obs, not_done, L, step)


        # Update decoder
        if self.decoder is not None and step % self.decoder_update_freq == 0:  # decoder_type is pixel
            decoder_update_dict = self.update_decoder(obs, action, next_obs, L, step)

        encoder_update_dict = {}
        transition_update_dict = {}
        if self.predict_inverse_dynamics:
            inverse_dynamics_loss, inverse_dynamics_dict = self.update_inverse_dynamics(obs, action, next_obs, L, step)
            total_loss = self.inverse_dynamics_loss_weight * inverse_dynamics_loss
            encoder_update_dict.update(inverse_dynamics_dict)
            self.inverse_dynamics_optimizer.zero_grad()
            total_loss.backward()
            self.inverse_dynamics_optimizer.step()

        # Update actor
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
            'actor_and_alpha': actor_and_alpha_update_dict,
            'decoder': decoder_update_dict
        }


    #     if self.decoder_type == 'contrastive':
    #         self.update_contrastive(obs, action, next_obs, L, step)
    #     elif self.decoder_type == 'inverse':
    #         self.update_inverse(obs, action, k_obs, L, step)



    # def update(self, replay_buffer, L, step):
    #     if self.decoder_type == 'inverse':
    #         obs, action, reward, next_obs, not_done, k_obs = replay_buffer.sample(k=True)
    #     else:
    #         obs, action, _, reward, next_obs, not_done = replay_buffer.sample()

    #     L.log('train/batch_reward', reward.mean(), step)

    #     self.update_critic(obs, action, reward, next_obs, not_done, L, step)

    #     if step % self.actor_update_freq == 0:
    #         self.update_actor_and_alpha(obs, L, step)

    #     if step % self.critic_target_update_freq == 0:
    #         utils.soft_update_params(
    #             self.critic.Q1, self.critic_target.Q1, self.critic_tau
    #         )
    #         utils.soft_update_params(
    #             self.critic.Q2, self.critic_target.Q2, self.critic_tau
    #         )
    #         utils.soft_update_params(
    #             self.critic.encoder, self.critic_target.encoder,
    #             self.encoder_tau
    #         )

    #     if self.decoder is not None and step % self.decoder_update_freq == 0:  # decoder_type is pixel
    #         self.update_decoder(obs, action, next_obs, L, step)

    #     if self.decoder_type == 'contrastive':
    #         self.update_contrastive(obs, action, next_obs, L, step)
    #     elif self.decoder_type == 'inverse':
    #         self.update_inverse(obs, action, k_obs, L, step)

    def to(self, device):
        self.device = device
        self.actor = self.actor.to(device=device)
        self.critic = self.critic.to(device=device)
        self.critic_target = self.critic_target.to(device=device)
        self.transition_model = self.transition_model.to(device=device)
        if self.decoder is not None:
            self.decoder = self.decoder.to(device=device)
        self.log_alpha = self.log_alpha.to(device=device)
        return self
    

    def state_dict(self, include_optimizers=True):
        """
        state dict consists of all constructor params needed to recreate model
        """
        constructor_params = {'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict(),
                            'critic_target': self.critic_target.state_dict(),
                            'transition_model': self.transition_model.state_dict(),
                            'log_alpha': self.log_alpha,
                            'target_entropy': self.target_entropy,
                            }
        if self.decoder is not None:
            constructor_params.update({
                'decoder': self.decoder.state_dict()
            })

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