from src.agent.bisim_agent_baseline import BisimAgent
import torch
import numpy as np
import torch.nn.functional as F

class KSMEBisimAgent(BisimAgent):
    """BisimAgent but with KSME encoder from https://arxiv.org/pdf/2310.19804.pdf"""
    def __init__(self, *args, **kwargs):
        """
        Need rew_min and rew_max to scale the kernel function
        """
        self.rew_min = kwargs.pop('rew_min', -1.)
        self.rew_max = kwargs.pop('rew_max', 1.)
        super().__init__(*args, **kwargs)

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
        reward2 = reward[perm]

        # Get the current kernel similarities
        z_dist = (h @ h2.t()).diag()

        # Get the target kernel similarities
        rew_scale = 1./(self.rew_max - self.rew_min) # if rew in [-1,1], this is 0.5
        reward_sim = 1. - rew_scale*F.smooth_l1_loss(reward, reward2, reduction='none').squeeze()
        with torch.no_grad():
            h_next = self.critic_target.encoder(next_obs)
            h2_next = h_next[perm]
        transition_sim = (h_next @ h2_next.t()).diag()
        ksme = reward_sim + self.discount*transition_sim

        loss = (z_dist - ksme).pow(2).mean()

        output_dict['encoder_loss'] = loss.item()

        return loss, output_dict
   

class NeuralEFKSMEBisimAgent(KSMEBisimAgent):

    def __init__(self, *args, **kwargs):
        self.normalize_kernel = kwargs.pop('normalize_kernel', False)
        super().__init__(*args, **kwargs)


    def update_encoder(self, obs, action, reward, L=None, step=None, next_obs=None):
        """
        Same as original but return output_dict in addition to the loss
        """
    
        h = self.critic.encoder(obs)            
        rew_scale = 1./(self.rew_max - self.rew_min) # if rew in [-1,1], this is 0.5
        reward_sim = 1. - rew_scale*torch.cdist(reward, reward, p=1)
        # Compute the kernel
        with torch.no_grad():
            h_next = self.critic_target.encoder(next_obs)
            transition_sim = (h_next @ h_next.t())
            kernel = reward_sim + self.discount*transition_sim

        loss, loss_dict = self._spectral_loss(kernel, h)

        return loss, loss_dict
    
    def _spectral_loss(self, W, features):
        """
        W : kernel matrix (weights matrix)
        """
        loss_dict = {}

        batch_size = W.shape[0]

        psis_x = features
        kernel = W

        if self.normalize_kernel:
            D_sqrt = torch.diag(torch.sum(kernel, dim=1)**(-0.5))
            #TODO: Make faster. Shouldn't waste time multiplying with diagonal matrix
            kernel = D_sqrt @ kernel @ D_sqrt

        K_psis = kernel @ psis_x
        psis_K_psis = psis_x.T @ K_psis
        psisSG_K_psis = psis_x.T.clone().detach() @ K_psis

        # The diagonal term
        loss_ii = torch.diag(psis_K_psis)
        # The off diagonal terms
        loss_ij = torch.triu(psisSG_K_psis, diagonal=1)**2
        loss_ij /= torch.diagonal(psis_K_psis.detach()).view(-1,1)
        loss_ij = loss_ij.sum(dim=0)

        loss = -(loss_ii - loss_ij).sum() / batch_size**2

        loss_dict['dist_loss'] = -loss_ii.detach().sum().item()/batch_size**2 # negative since we're minimizing
        loss_dict['ortho_loss'] =  loss_ij.detach().sum().item()/batch_size**2 # positive since we're minimizing
        loss_dict['encoder_loss'] = loss.detach().item()

        return loss, loss_dict
