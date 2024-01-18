import numpy as np
from types import SimpleNamespace
from src.agent.bisim_agent_baseline import BisimAgent
from src.agent.bisim_agent_spectral import SpectralBisimAgent, NeuralEFBisimAgent
from src.agent.bisim_agent_ksme import KSMEBisimAgent, NeuralEFKSMEBisimAgent
from src.agent.sac_agent import SACAgent
from src.agent.rap_agent import RAPBisimAgent, NeuralEFRAPBisimAgent


DISTANCE_TYPES = {'dbc': 'bisim', 'mico': 'mico'}

def make_agent(obs_shape, action_shape, args, device):
    
    if isinstance(args, dict):
        args = SimpleNamespace(**args)

    if (args.agent is None) or (args.agent == np.nan):
        raise ValueError("args.agent is None or NaN")
    
    agent_kwargs = {
        'obs_shape': obs_shape,
        'action_shape': action_shape,
        'device': device,
        'hidden_dim': args.hidden_dim,
        'discount': args.discount,
        'init_temperature': args.init_temperature,
        'alpha_lr': args.alpha_lr,
        'alpha_beta': args.alpha_beta,
        'actor_lr': args.actor_lr,
        'actor_beta': args.actor_beta,
        'actor_log_std_min': args.actor_log_std_min,
        'actor_log_std_max': args.actor_log_std_max,
        'actor_update_freq': args.actor_update_freq,
        'critic_lr': args.critic_lr,
        'critic_beta': args.critic_beta,
        'critic_tau': args.critic_tau,
        'critic_target_update_freq': args.critic_target_update_freq,
        'encoder_type': args.encoder_type,
        'encoder_feature_dim': args.encoder_feature_dim,
        'encoder_lr': args.encoder_lr,
        'encoder_tau': args.encoder_tau,
        'encoder_stride': args.encoder_stride,
        'decoder_type': args.decoder_type,
        'decoder_lr': args.decoder_lr,
        'decoder_update_freq': args.decoder_update_freq,
        'decoder_weight_lambda': args.decoder_weight_lambda,
        'transition_model_type': args.transition_model_type,
        'num_layers': args.num_layers,
        'num_filters': args.num_filters,
        'bisim_coef': args.bisim_coef,
        'decode_rewards_from_next_latent': args.decode_rewards_from_next_latent,
        'residual_actor': args.residual_actor,
        'use_schedulers': args.use_schedulers,
        'encoder_softmax': args.encoder_softmax,
        'distance_type': DISTANCE_TYPES.get(args.encoder_mode),
        'predict_inverse_dynamics': args.predict_inverse_dynamics,
        'inverse_dynamics_lr': args.inverse_dynamics_lr,
        'inverse_dynamics_loss_weight': args.inverse_dynamics_loss_weight,
        'encoder_max_norm': args.encoder_max_norm,
        'intrinsic_reward': args.intrinsic_reward,
        'intrinsic_reward_max': args.intrinsic_reward_max,
        'intrinsic_reward_scale': args.intrinsic_reward_scale,
        'trunk_regularization': args.trunk_regularization,
        'trunk_regularization_coeff': args.trunk_regularization_coeff
    }

    
    if args.agent == 'bisim':
        if args.encoder_mode in ['dbc', 'mico']:
            # Baseline Bisim Agent
            agent = BisimAgent(**agent_kwargs)
        elif args.encoder_mode == 'spectral':
            # Spectral Bisim Agent
            agent_kwargs.update({
                'encoder_kernel_bandwidth': args.encoder_kernel_bandwidth,
                'encoder_normalize_loss': args.encoder_normalize_loss,
                'encoder_ortho_loss_reg': args.encoder_ortho_loss_reg,
            })
            agent = SpectralBisimAgent(**agent_kwargs)
        elif args.encoder_mode == 'neural_ef':
            agent_kwargs.update({'normalize_kernel': args.normalize_kernel})
            agent = NeuralEFBisimAgent(**agent_kwargs)
        elif args.encoder_mode == 'ksme':
            agent_kwargs.update({'rew_max': args.rew_max, 'rew_min': args.rew_min })
            agent = KSMEBisimAgent(**agent_kwargs)
        elif args.encoder_mode == 'neural_ef_ksme':
            agent_kwargs.update({'normalize_kernel': args.normalize_kernel, 
                                 'rew_max': args.rew_max, 
                                 'rew_min': args.rew_min,
                                 'kernel_type': args.kernel_type
                                 })
            agent = NeuralEFKSMEBisimAgent(**agent_kwargs)
        elif args.encoder_mode == 'rap':
            agent_kwargs.update({
                'rap_structural_distance': args.rap_structural_distance,
                'rap_reward_dist': args.rap_reward_dist,
                'rap_square_target': args.rap_square_target,
            })
            agent = RAPBisimAgent(**agent_kwargs)
        elif args.encoder_mode == 'neural_ef_rap':
            agent_kwargs.update({
                'rap_structural_distance': args.rap_structural_distance,
                'rap_reward_dist': args.rap_reward_dist,
                'rap_square_target': args.rap_square_target,
                'normalize_kernel': args.normalize_kernel,
                'kernel_type': args.kernel_type
            })
            agent = NeuralEFRAPBisimAgent(**agent_kwargs)
        else:
            raise ValueError(f"Unknown encoder_mode {args.encoder_mode}")
    elif args.agent == 'sac':
        agent = SACAgent(**agent_kwargs)
    else:
        raise NotImplementedError(f"Agent {args.agent} not implemented")
    return agent