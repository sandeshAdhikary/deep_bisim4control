from src.logger import Logger

def setup_logger(args):
    if args.logger == 'tensorboard':
        logger_config = {'log_dir': args.log_dir,
                            'sw': 'tensorboard', 
                            'format_config': 'rl'
                            }
    elif args.logger == 'wandb':
        logger_config = {'log_dir': args.log_dir,
                            'sw': 'wandb',
                            'project': args.logger_project,
                            'tracked_params': args.__dict__
                         }
    else:
        raise ValueError("Unknown Logger")
    return Logger(logger_config)