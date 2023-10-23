
from trainer import Logger, Model, Sweeper
from trainer.rl import RLTrainer

# Project specific imports
from src.train.setup_env import make_env
from src.train.make_agent import make_agent

from typing import Dict
import os
from copy import copy
import yaml
import wandb


class BisimModel(Model):

    def __init__(self, config: Dict):
        """
        Define self.model here
        """
        super().__init__(config)

        model_config = copy(config)
        self.model = make_agent(
            obs_shape=model_config.pop('obs_shape'),
            action_shape=model_config.pop('action_shape'),
            device=model_config.get('device'),
            args=model_config,
        )

    @property
    def module_path(self):
        return 'src.sweep'
        
    def parse_config(self, config):

        # Set encoder output dim to be feature dim if not set
        if not isinstance(config['encoder_output_dim'], int):
            config['encoder_output_dim'] = config['encoder_feature_dim']

        # If using cluster encoders, num_rews is the same as encoder's output dim
        if 'cluster' in config['encoder_type']:
            config['reward_decoder_num_rews'] = config['encoder_output_dim']

        return config

    def training_step(self, batch, batch_idx, step):
        return self.model.update(batch, step)
    
    def save_model(self, filename, save_optimizers=True):
        self.model.save(model_dir=os.path.dirname(filename),
                        filename=os.path.basename(filename),
                        save_optimizers=save_optimizers
                        )

    def load_model(self, model_file=None, model_dir=None, chkpt_name=None):
        if model_file is not None:
            self.model.load(model_file=model_file)
        else:
            chkpt_name = chkpt_name or 'eval_checkpoint'
            self.model.load(model_dir=model_dir, step=chkpt_name)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def zero_grad(self):
        self.model.zero_grad()

    def sample_action(self, obs, batched=False):
        """
        Sample action from model. May be non-deterministic
        """
        return self.model.sample_action(obs, batched=batched)

    def select_action(self, obs, batched=False):
        """
        Select action from model. Should be deterministic
        """
        return self.model.select_action(obs, batched=batched)


class BisimRLTrainer(RLTrainer):

    @property
    def module_path(self):
        return 'src.sweep'

    def make_env(self, config):
        """
        Return an env given a config
        """
        return make_env(config)
    
    def train_step(self, batch, batch_idx=None):
        """
        Overwriting parent method to add 'step' argument to model.training_step
        """
        # The model's training step also needs access to trainer.step
        train_step_output = self.model.training_step(batch, batch_idx, step=self.step)
        self.train_log.append({'step': self.step, 'log': train_step_output})

    def parse_config(self, config: Dict):

        if config.get('img_source') in ['none', 'None']:
            config['env']['img_source'] = None

        return config
    
    def log_epoch(self, info: Dict):
        super().log_epoch(info)
        if len(self.train_log) >= 1:
            # Log metrics from train_log
            last_log = self.train_log[-1]['log']
            trainer_step = self.train_log[-1]['step']
            self.logger.log(log_dict={'trainer_step': trainer_step,
                                      'train/critic/loss':last_log['critic']['loss']})
            if last_log['actor_and_alpha'] is not None:
                self.logger.log(log_dict={'trainer_step': trainer_step,
                                        'train/actor/loss':last_log['actor_and_alpha']['loss']})
                self.logger.log(log_dict={'trainer_step': trainer_step,
                                        'train/actor/target_entropy':last_log['actor_and_alpha']['target_entropy']})
                self.logger.log(log_dict={'trainer_step': trainer_step,
                                        'train/actor/entropy':last_log['actor_and_alpha']['entropy']})
                self.logger.log(log_dict={'trainer_step': trainer_step,
                                        'train/alpha/loss':last_log['actor_and_alpha']['alpha_loss']})
                self.logger.log(log_dict={'trainer_step': trainer_step,
                                        'train/alpha/value':last_log['actor_and_alpha']['alpha_value']})
            self.logger.log(log_dict={'trainer_step': trainer_step,
                                      'train/encoder/loss':last_log['encoder']['encoder_loss']})
            ortho_loss = last_log['encoder'].get('ortho_loss')
            if ortho_loss is not None:
                self.logger.log(log_dict={'trainer_step': trainer_step,
                                        'train/encoder/ortho_loss':ortho_loss})
            dist_loss = last_log['encoder'].get('dist_loss')
            if dist_loss is not None:
                self.logger.log(log_dict={'trainer_step': trainer_step,
                                        'train/encoder/dist_loss':dist_loss})
        if len(self.eval_log) >= 1:
            last_log = self.eval_log[-1]['log']
            trainer_step = self.eval_log[-1]['step']
            # Log individiaul episode rewards
            avg_env_episode_rewards = last_log.get('avg_env_episode_rewards')
            if (avg_env_episode_rewards is not None) and len(avg_env_episode_rewards) > 1:
                for ide, env_rew in enumerate(avg_env_episode_rewards):
                    self.logger.log(log_dict={'eval_step': int(trainer_step),
                                              f'eval/episode_reward/env_{ide}': float(env_rew)})                
                    
            # Get obs video
            obs_video = last_log.get('obs_video')

            if obs_video is not None:
                #TODO: Need to set up obs_video to take in dict
                pass

def objective(project_name=None, default_params=None, run_id=None, sweep_callback=None):
    run = None
    if run_id is not None:
        run = wandb.init(project=project_name, 
                        id=run_id, 
                        resume='must',
                        dir=default_params['logger']['dir'])

    # Instantiate trainer
    trainer = BisimRLTrainer(default_params['trainer'])
    # Set up model
    default_params['model'].update(trainer.env.get_env_shapes())
    model = BisimModel(default_params['model'])
    trainer.set_model(model)
    # Set up logger
    logger = Logger(default_params['logger'], run=run)
    trainer.set_logger(logger)
    # Train
    trainer.fit()


if __name__ == "__main__":

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str)
    parser.add_argument('--num_runs', type=int, default=1)
    args = parser.parse_args()  

    config = yaml.safe_load(open(args.config, 'r'))

    sweeper = Sweeper(config, BisimRLTrainer, BisimModel, Logger)

    sweeper.sweep(count=args.num_runs)