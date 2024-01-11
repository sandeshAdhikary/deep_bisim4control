from trainer.rl import RLTrainer
from typing import Dict
from trainer.utils import eval_mode

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
            if last_log['critic'].get('trunk_reg') is not None:
                log_dict.update({
                    'train/critic/trunk_reg': last_log['critic']['trunk_reg']
                })

            # Log actor and alpha
            if last_log['actor_and_alpha'] is not None:
                log_dict.update({
                    'train/actor/loss':last_log['actor_and_alpha']['loss'],
                    'train/actor/target_entropy':last_log['actor_and_alpha']['target_entropy'],
                    'train/actor/entropy':last_log['actor_and_alpha']['entropy'],
                    'train/alpha/loss':last_log['actor_and_alpha']['alpha_loss'],
                    'train/alpha/value':last_log['actor_and_alpha']['alpha_value']
                    })
                if last_log['actor_and_alpha'].get('trunk_reg') is not None:
                    log_dict.update({
                        'train/actor/trunk_reg': last_log['actor_and_alpha']['trunk_reg']
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


            # log embedding norms
            embedding_norm_log = last_log['encoder'].get('embedding_norm')
            if embedding_norm_log is not None:
                log_dict.update({
                    'train/encoder/norm': embedding_norm_log
                })

            # Log inverse dynamics logs
            inverse_dynamics_log = last_log['encoder'].get('inverse_dynamics_loss')
            if inverse_dynamics_log is not None:
                log_dict.update({
                    'train/encoder/inverse_dynamics_loss': inverse_dynamics_log
                })


            self.logger.log(log_dict=log_dict)

            # Log eigenvalues
            if last_log['encoder'].get('eigenvalues') is not None:
                self.logger.log_linechart(
                    key = 'train/encoder/eigenvalues',
                    data = {
                        'title': 'Eigenvalues',
                        'x': [range(len(last_log['encoder']['eigenvalues']))],
                        'y': [last_log['encoder']['eigenvalues']],
                    }
                )

        if len(self.eval_log) >= 1:
            last_log = self.eval_log[-1]['log']
            trainer_step = self.eval_log[-1]['step']
            # Log individiaul episode rewards
            avg_env_episode_rewards = last_log.get('avg_env_episode_rewards')
            if (avg_env_episode_rewards is not None) and len(avg_env_episode_rewards) > 1:
                for ide, env_rew in enumerate(avg_env_episode_rewards):
                    self.logger.log(log_dict={'eval_step': int(trainer_step),
                                              f'eval/episode_reward/env_{ide}': float(env_rew)})
                    

    def collect_rollouts(self, obs, add_to_buffer=False):
        # Set model to eval when collecting rollouts
        with eval_mode(self.model):
            # Get Action
            if self.step < self.config['init_steps']:
                action = self.env.action_space.sample()
            else:
                action = self.model.sample_action(obs, batched=self.env.is_vec_env)
        
            # Env Step
            next_obs, reward, terminated, truncated, info = self.env.step(action)
            curr_reward = self.reward

            # Optionally allow modifying experience (e.g. adding auxilliary rewards)
            obs, action, curr_reward, reward, next_obs, terminated, truncated, info = self.modify_experience(
                obs, action, curr_reward, reward, next_obs, terminated, truncated, info
            )
            if add_to_buffer:
                # TODO: Move this to after_epoch()
                # Add to buffer: allow infinite bootstrap: don't store truncated as done    
                self.replay_buffer.add(obs, action, curr_reward, reward, next_obs, terminated, 
                                    batched=self.env.is_vec_env)

            num_steps = self.env.num_envs

        return {
            'action': action,
            'next_obs': next_obs,
            'reward': reward,
            'terminated': terminated,
            'truncated': truncated,
            'info': info,
            'num_steps': num_steps
        }
    
    def modify_experience(self, obs, action, curr_reward, reward, next_obs, terminated, truncated, info):

        if hasattr(self.model, 'modify_experience'):
            obs, action, curr_reward, reward, next_obs, terminated, truncated, info = self.model.modify_experience(
               obs, action, curr_reward, reward, next_obs, terminated, truncated, info
            )

        return obs, action, curr_reward, reward, next_obs, terminated, truncated, info