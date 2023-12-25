import os
from typing import Dict
from trainer import Model
from copy import deepcopy
from trainer.utils import import_module_attr

class BisimModel(Model):

    def __init__(self, config: Dict):
        """
        Define self.model here
        """
        # Import function to make agent
        self.make_agent = import_module_attr(config['make_agent_module_path'])

        model_config = deepcopy(config)
        model = self.make_agent(
            obs_shape=model_config.pop('obs_shape'),
            action_shape=model_config.pop('action_shape'),
            device=model_config.get('device'),
            args=model_config,
        )

        super().__init__(config, model=model, optimizer=None, loss_fn=None)
    # @property
    # def module_path(self):
    #     return 'src.study.models'
        
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
    
    def state_dict(self, **kwargs):
        return self.model.state_dict(**kwargs)
    
    def save_model(self, filename, save_optimizers=True):
        self.model.save(model_dir=os.path.dirname(filename),
                        filename=os.path.basename(filename),
                        save_optimizers=save_optimizers
                        )

    def load_model(self, state_dict=None, model_file=None, model_dir=None, chkpt_name=None):
        if state_dict is not None:
            self.model.load(state_dict=state_dict)
        elif model_file is not None:
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
    

    def evaluation_step(self, batch, batch_idx):
        return {}