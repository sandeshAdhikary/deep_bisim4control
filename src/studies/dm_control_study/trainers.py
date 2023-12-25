from trainer.rl import RLTrainer
from typing import Dict

class BisimRLTrainer(RLTrainer):

    @property
    def module_path(self):
        return 'src.study.trainers'
    
    def _setup_metric_loggers(self, config, metrics=None):
        pass
    
    def train_step(self, batch, batch_idx=None):
        """
        Overwriting parent method to add 'step' argument to model.training_step
        """
        # The model's training step also needs access to trainer.step
        train_step_output = self.model.training_step(batch, batch_idx, step=self.model_update_steps)
        self.train_log.append({'step': self.step, 'log': train_step_output})
        self.model_update_steps += 1

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

            # Log critic
            log_dict = {
                'trainer_step': trainer_step,
                'train/critic/loss':last_log['critic']['loss']
                }
            # Log actor and alpha
            if last_log['actor_and_alpha'] is not None:
                log_dict.update({
                    'train/actor/loss':last_log['actor_and_alpha']['loss'],
                    'train/actor/target_entropy':last_log['actor_and_alpha']['target_entropy'],
                    'train/actor/entropy':last_log['actor_and_alpha']['entropy'],
                    'train/alpha/loss':last_log['actor_and_alpha']['alpha_loss'],
                    'train/alpha/value':last_log['actor_and_alpha']['alpha_value']
                    })
            # log encoder
            log_dict['train/encoder/loss'] = last_log['encoder']['encoder_loss']
            ortho_loss = last_log['encoder'].get('ortho_loss')
            if ortho_loss is not None:
                log_dict['train/encoder/ortho_loss'] = ortho_loss
            dist_loss = last_log['encoder'].get('dist_loss')
            if dist_loss is not None:
                log_dict['train/encoder/dist_loss'] = dist_loss

            # Log learning rates
            log_dict.update({
                'train/critic/lr': self.model.model.critic_optimizer.param_groups[0]['lr'],
                'train/actor/lr': self.model.model.actor_optimizer.param_groups[0]['lr'],
                'train/alpha/lr': self.model.model.log_alpha_optimizer.param_groups[0]['lr'],
                'train/encoder/lr': self.model.model.encoder_optimizer.param_groups[0]['lr']
                })
            self.logger.log(log_dict=log_dict)


            # self.logger.log(log_dict={'trainer_step': trainer_step,
            #                           'train/critic/loss':last_log['critic']['loss']})
            # if last_log['actor_and_alpha'] is not None:
            #     self.logger.log(log_dict={'trainer_step': trainer_step,
            #                             'train/actor/loss':last_log['actor_and_alpha']['loss']})
            #     self.logger.log(log_dict={'trainer_step': trainer_step,
            #                             'train/actor/target_entropy':last_log['actor_and_alpha']['target_entropy']})
            #     self.logger.log(log_dict={'trainer_step': trainer_step,
            #                             'train/actor/entropy':last_log['actor_and_alpha']['entropy']})
            #     self.logger.log(log_dict={'trainer_step': trainer_step,
            #                             'train/alpha/loss':last_log['actor_and_alpha']['alpha_loss']})
            #     self.logger.log(log_dict={'trainer_step': trainer_step,
            #                             'train/alpha/value':last_log['actor_and_alpha']['alpha_value']})
            # self.logger.log(log_dict={'trainer_step': trainer_step,
            #                           'train/encoder/loss':last_log['encoder']['encoder_loss']})
            # ortho_loss = last_log['encoder'].get('ortho_loss')
            # if ortho_loss is not None:
            #     self.logger.log(log_dict={'trainer_step': trainer_step,
            #                             'train/encoder/ortho_loss':ortho_loss})
            # dist_loss = last_log['encoder'].get('dist_loss')
            # if dist_loss is not None:
            #     self.logger.log(log_dict={'trainer_step': trainer_step,
            #                             'train/encoder/dist_loss':dist_loss})
        if len(self.eval_log) >= 1:
            last_log = self.eval_log[-1]['log']
            trainer_step = self.eval_log[-1]['step']
            # Log individiaul episode rewards
            avg_env_episode_rewards = last_log.get('avg_env_episode_rewards')
            if (avg_env_episode_rewards is not None) and len(avg_env_episode_rewards) > 1:
                for ide, env_rew in enumerate(avg_env_episode_rewards):
                    self.logger.log(log_dict={'eval_step': int(trainer_step),
                                              f'eval/episode_reward/env_{ide}': float(env_rew)})