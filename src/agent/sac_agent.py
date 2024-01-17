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
from src.agent.bisim_agent_baseline import BisimAgent

class SACAgent(BisimAgent):
    """
    SAC Agent
    Same as BisimAgent, but we remove all encoder, decoder, transition model updates
    i.e. only actor and critic updates remain
    """
    
    def _update(self, obs, action, reward, next_obs, not_done, step, L=None):
        """
        Same as _update for BisimAgent. But removing encoder/decoder/transition_model updates
        """

        critic_update_dict = self.update_critic(obs, action, reward, next_obs, not_done, L, step)

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
        return {
            'critic': critic_update_dict,
            'actor_and_alpha': actor_and_alpha_update_dict
        }

    def state_dict(self, include_optimizers=True):
        """
        state dict consists of all constructor params needed to recreate model
        """
        constructor_params = {'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict(),
                            'critic_target': self.critic_target.state_dict(),
                            'log_alpha': self.log_alpha,
                            'target_entropy': self.target_entropy,
                            }

        if include_optimizers:
            constructor_params.update({
                'actor_optimizer': self.actor_optimizer.state_dict(),
                'critic_optimizer': self.critic_optimizer.state_dict(),
                'log_alpha_optimizer': self.log_alpha_optimizer.state_dict(),
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
        self.log_alpha = self.log_alpha.to(device=device)
        return self
    