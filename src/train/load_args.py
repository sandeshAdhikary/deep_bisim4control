import argparse
from types import SimpleNamespace
import os

def parse_args(args=None):


    if args is not None:
        # Load from the given args
        args = SimpleNamespace(**args)
    else:
        # Load from CLI inputs

        parser = argparse.ArgumentParser()
        parser.add_argument('--config', default=None, type=str, 
                            help="Path to hyperparams config file. All other args are ignored when config is provided")
        # random seeds
        parser.add_argument('--seed', default=1, type=int)
        parser.add_argument('--num_seeds', default=1, type=int)
        # environment
        parser.add_argument('--domain_name', default='cheetah')
        parser.add_argument('--task_name', default='run')
        parser.add_argument('--image_size', default=88, type=int)
        parser.add_argument('--action_repeat', default=1, type=int)
        parser.add_argument('--frame_stack', default=3, type=int)
        parser.add_argument('--resource_files', type=str)
        parser.add_argument('--eval_resource_files', type=str)
        parser.add_argument('--img_source', default=None, type=str, choices=['color', 'noise', 'images', 'video', 'none', 'mnist', 'driving_stereo'])
        parser.add_argument('--total_frames', default=1000, type=int)
        parser.add_argument('--distractor', default='None', choices=['ideal_gas', "None"])
        parser.add_argument('--distractor_type', default='None', choices=['overlay', 'padding'])
        parser.add_argument('--distractor_img_shrink_factor', default=1.3, type=float)
        parser.add_argument('--distraction_level', default=0.2, type=float)
        parser.add_argument('--boxed_env', default=False, action='store_true')
        parser.add_argument('--episode_length', default=1_000, type=int)
        # replay buffer
        parser.add_argument('--replay_buffer_capacity', default=10_000, type=int)
        # train
        parser.add_argument('--agent', default='bisim', type=str, choices=['baseline', 'bisim', 'deepmdp', 'bisim_decomp'])
        parser.add_argument('--init_steps', default=1000, type=int)
        parser.add_argument('--num_train_steps', default=1000000, type=int)
        parser.add_argument('--batch_size', default=512, type=int)
        parser.add_argument('--hidden_dim', default=256, type=int)
        parser.add_argument('--k', default=3, type=int, help='number of steps for inverse model')
        parser.add_argument('--bisim_coef', default=0.5, type=float, help='coefficient for bisim terms')
        parser.add_argument('--load_encoder', default=None, type=str)
        parser.add_argument('--num_train_envs', default=1, type=int)
        parser.add_argument('--agent_load_path', default=None, type=str)
        # eval
        parser.add_argument('--num_eval_envs', default=1, type=int)
        parser.add_argument('--eval_freq', default=10, type=int)  # TODO: master had 10000
        parser.add_argument('--num_eval_episodes', default=20, type=int)
        # critic
        parser.add_argument('--critic_lr', default=1e-3, type=float)
        parser.add_argument('--critic_beta', default=0.9, type=float)
        parser.add_argument('--critic_tau', default=0.005, type=float)
        parser.add_argument('--critic_target_update_freq', default=2, type=int)
        parser.add_argument('--use_cagrad', default=False, action='store_true')
        parser.add_argument('--vec_reward_from_model', default=False, action='store_true')
        parser.add_argument('--reward_decomp_method', default='eigenrewards', choices=['eigenrewards', 'cluster'])
        # actor
        parser.add_argument('--actor_lr', default=1e-3, type=float)
        parser.add_argument('--actor_beta', default=0.9, type=float)
        parser.add_argument('--actor_log_std_min', default=-10, type=float)
        parser.add_argument('--actor_log_std_max', default=2, type=float)
        parser.add_argument('--actor_update_freq', default=2, type=int)
        # encoder/decoder
        parser.add_argument('--encoder_type', default='pixel', type=str, choices=[
            'pixel', 'pixelCarla096', 'pixelCarla098', 'identity', 'vector',
            'pixel_cluster', 'pixelCarla096_cluster', 'pixelCarla098_cluster', 'identity_cluster', 'vector_cluster'
            ])
        parser.add_argument('--encoder_feature_dim', default=50, type=int)
        parser.add_argument('--encoder_output_dim', default=None, type=int)
        parser.add_argument('--encoder_lr', default=1e-3, type=float)
        parser.add_argument('--encoder_tau', default=0.005, type=float)
        parser.add_argument('--encoder_stride', default=1, type=int)
        parser.add_argument('--decoder_type', default='pixel', type=str, choices=['pixel', 'identity', 'contrastive', 'reward', 'inverse', 'reconstruction'])
        parser.add_argument('--decoder_lr', default=1e-3, type=float)
        parser.add_argument('--decoder_update_freq', default=1, type=int)
        parser.add_argument('--decoder_weight_lambda', default=0.0, type=float)
        parser.add_argument('--num_layers', default=4, type=int)
        parser.add_argument('--num_filters', default=32, type=int)
        parser.add_argument('--encoder_mode', default='spectral', choices=['spectral', 'dbc'])
        parser.add_argument('--encoder_kernel_bandwidth', default='auto')
        parser.add_argument('--encoder_normalize_loss', default=True, action='store_true')
        parser.add_argument('--encoder_ortho_loss_reg', default=1e-4, type=float)
        parser.add_argument('--reward_decoder_num_rews', default=1, type=int)
        # sac
        parser.add_argument('--discount', default=0.99, type=float)
        parser.add_argument('--init_temperature', default=0.01, type=float)
        parser.add_argument('--alpha_lr', default=1e-3, type=float)
        parser.add_argument('--alpha_beta', default=0.9, type=float)
        # misc
        parser.add_argument('--save_tb', default=False, action='store_true')
        parser.add_argument('--save_model', default=False, action='store_true')
        parser.add_argument('--save_buffer', default=False, action='store_true')
        parser.add_argument('--save_video', default=False, action='store_true')
        parser.add_argument('--transition_model_type', default='', type=str, choices=['', 'deterministic', 'probabilistic', 'ensemble'])
        parser.add_argument('--render', default=False, action='store_true')
        parser.add_argument('--port', default=2000, type=int)
        # Evaluation
        parser.add_argument('--eval_img_sources', default=None, type=str)
        # Logging
        parser.add_argument('--logger', default='tensorboard', type=str, choices=['tensorboard', 'wandb'])
        parser.add_argument('--logger_project', default='misc', type=str)
        parser.add_argument('--log_dir', default='/project/logdir', type=str)
        parser.add_argument('--logger_img_downscale_factor', default=3, type=int)
        parser.add_argument('--logger_video_log_freq', default=None, type=int)
        parser.add_argument('--logger_tags', default=None, type=str)
        parser.add_argument('--logger_minimal', default=False, action='store_true')
        # level set experiment args
        parser.add_argument('--levelset_factor', default=1.0, type=float)
        # Sweep config
        parser.add_argument('--sweep_config', default=None, type=str)
        args = parser.parse_args()
        
        if args.config is not None:
            args = args_from_config(args.config)
    
    # Some minor updates to args
    args = update_args(args)

    return args


def args_from_config(config_path):

    import yaml
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    if config.get('project_name') is not None:
        config['logger_project'] = config.pop('project_name')
    else:
        assert config['logger_project'] is not None

    args = SimpleNamespace(**config)

    return args

def update_args(args):

    
    # Convert eval_img_sources from a single list to a list of strings
    if isinstance(args.eval_img_sources, str):
        args.eval_img_sources = args.eval_img_sources.strip("(')").replace("'", "")
        args.eval_img_sources = args.eval_img_sources.replace("[", "")
        args.eval_img_sources = args.eval_img_sources.replace("]", "")
        args.eval_img_sources = [item.strip() for item in args.eval_img_sources.split(',')]


    if hasattr(args, 'logger_tags') and isinstance(args.logger_tags, str):
        args.logger_tags = args.logger_tags.strip("(')").replace("'", "")
        args.logger_tags = args.logger_tags.replace("[", "")
        args.logger_tags = args.logger_tags.replace("]", "")
        args.logger_tags = [item.strip() for item in args.logger_tags.split(',')]


    # Set encoder output dim to be feature dim if not set
    if not isinstance(args.encoder_output_dim, int):
        args.encoder_output_dim = args.encoder_feature_dim

    # If using cluster encoders, num_rews is the same as encoder's output dim
    if 'cluster' in args.encoder_type:
        args.reward_decoder_num_rews = args.encoder_output_dim

    if args.img_source == 'none':
        args.img_source = None

    # Set logger_video_log_freq
    if args.logger_video_log_freq in [None, 'none', 'None']:
        # Set logger_video_log_freq so we get max 5 videos per run
        num_video_logs = 5
        num_evals = int(args.num_train_steps // args.eval_freq)
        args.logger_video_log_freq = max(int(num_evals / num_video_logs), 1)

    # TODO: Set run name

    return args