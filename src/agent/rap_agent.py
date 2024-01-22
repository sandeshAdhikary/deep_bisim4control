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
from src.agent.bisim_agent_baseline import BisimAgent
from sklearn.metrics.pairwise import cosine_similarity as pairwise_consine_similarity



class RAPBisimAgent(BisimAgent):
    """
    RAP Agent
    """
    # EPSILON = 1e-9 # original epsilon from the RAP repo; resulted in nans
    # # EPSILON = 1e-5 # New epsilon to avoid nans

    def __init__(self, *args, **kwargs):
        self.rap_structural_distance = kwargs.pop('rap_structural_distance', 'l1_smooth')
        self.rap_reward_dist = kwargs.pop('rap_reward_dist', False)
        self.rap_square_target = kwargs.pop('rap_square_target', False)
        self.epsilon = kwargs.pop('epsilon', 1e-9) # noise used to stabilize sqrt and denominators
        super().__init__(*args, **kwargs)


        encoder_feature_dim = kwargs.get('encoder_feature_dim')
        self.state_reward_decoder = StateRewardDecoder(
            encoder_feature_dim).to(self.device)


        # Re-initialize decoder_optimizer, this time with state_reward_decoder as well
        decoder_lr = kwargs.get('decoder_lr')
        decoder_weight_lambda = kwargs.get('decoder_weight_lambda')
        self.decoder_optimizer = torch.optim.Adam(
            list(self.reward_decoder.parameters()) + list(self.transition_model.parameters()) + list(self.state_reward_decoder.parameters()),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        self.train()
        self.critic_target.train()

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

        with torch.no_grad():
            # action, _, _, _ = self.actor(obs, compute_pi=False, compute_log_pi=False)
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            # reward = self.reward_decoder(pred_next_latent_mu1)
            reward2 = reward[perm]
        if pred_next_latent_sigma1 is None:
            pred_next_latent_sigma1 = torch.zeros_like(pred_next_latent_mu1)
        if pred_next_latent_mu1.ndim == 2:  # shape (B, Z), no ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[perm]
        elif pred_next_latent_mu1.ndim == 3:  # shape (B, E, Z), using an ensemble
            pred_next_latent_mu2 = pred_next_latent_mu1[:, perm]
            pred_next_latent_sigma2 = pred_next_latent_sigma1[:, perm]
        else:
            raise NotImplementedError

        z_dist = self.metric_func(h, h2)
        
        if self.rap_reward_dist:
            reward_mu, reward_sigma = self.state_reward_decoder(h)
            loss_reward_decoder = self.state_reward_decoder.loss(
                reward_mu, reward_sigma, reward
            )

        if self.rap_reward_dist:
            reward_var = reward_sigma.detach().pow(2.)
            reward_var2 = reward_var[perm]
            r_var = reward_var
            r_var2 = reward_var2

            reward_mu2 = reward_mu[perm]
            r_mean = reward_mu
            r_mean2 = reward_mu2

        with torch.no_grad():
            if self.rap_reward_dist:
                r_dist = (reward - reward2).pow(2.)
                r_dist = F.relu(r_dist - r_var - r_var2)
                r_dist = self._sqrt(r_dist)
            else:
                r_dist = F.smooth_l1_loss(reward, reward2, reduction='none')

        if self.transition_model_type == '':
            transition_dist = F.smooth_l1_loss(pred_next_latent_mu1, pred_next_latent_mu2, reduction='none')
        else:
            if self.rap_structural_distance == 'x^2+y^2-xy' or self.rap_structural_distance == 'mico_angular':
                transition_dist = self.metric_func(pred_next_latent_mu1, pred_next_latent_mu2)
            else:
                transition_dist = torch.sqrt(
                    (pred_next_latent_mu1 - pred_next_latent_mu2).pow(2) +
                    (pred_next_latent_sigma1 - pred_next_latent_sigma2).pow(2)
                ).mean(dim=-1, keepdim=True)
            
        if self.rap_square_target:
            assert self.rap_reward_dist
            with torch.no_grad():
                r_dist_square = (reward - reward2).pow(2.)
                r_dist_square_minus_var = r_dist_square - r_var - r_var2
            diff_square = (z_dist - self.discount * transition_dist).pow(2.)
            loss = F.smooth_l1_loss(diff_square, r_dist_square_minus_var, reduction='mean')
        else:
            rap_dist_target = r_dist + self.discount * transition_dist
            loss = F.smooth_l1_loss(z_dist, rap_dist_target, reduction='mean')
        if self.rap_reward_dist:
            loss = loss + loss_reward_decoder


        output_dict['encoder_loss'] = loss.item()
        if L is not None:
            L.log('train_ae/encoder_loss', loss, step)

        output_dict['embedding_norm'] = torch.norm(h, dim=1).mean().item()
        output_dict['state_reward_decoder_loss'] = loss_reward_decoder.item()

        return loss, output_dict


    def metric_func(self, x, y):
        if self.rap_structural_distance == 'l2':
            dist = F.pairwise_distance(x, y, p=2, keepdim=True)
        elif self.rap_structural_distance == 'l1_smooth':
            dist = F.smooth_l1_loss(x, y, reduction='none')
            dist = dist.mean(dim=-1, keepdim=True)
        elif self.rap_structural_distance == 'mico_angular':
            beta = 1e-6 #1e-5 # #0.1
            base_distances = self._cosine_distance(x, y)
            # print("base_distances", base_distances)
            norm_average = (x.pow(2.).sum(dim=-1, keepdim=True)
            # norm_average = 0.5 * (x.pow(2.).sum(dim=-1, keepdim=True) 
                + y.pow(2.).sum(dim=-1, keepdim=True))
            dist = norm_average + beta * base_distances
        elif self.rap_structural_distance == 'x^2+y^2-xy':
            # beta = 1.0 # 0 < beta < 2
            k = 0.1 # 0 < k < 2
            base_distances = (x * y).sum(dim=-1, keepdim=True)
            # print("base_distances", base_distances)
            norm_average = (x.pow(2.).sum(dim=-1, keepdim=True) 
                + y.pow(2.).sum(dim=-1, keepdim=True))
            # dist = norm_average - (2. - beta) * base_distances
            dist = norm_average - k * base_distances
            # dist = dist.sqrt()
        else:
            raise NotImplementedError
        return dist


    def state_dict(self, include_optimizers=True):
        """
        Update constructor params with state_reward_decoder
        """
        constructor_params = {'actor': self.actor.state_dict(),
                            'critic': self.critic.state_dict(),
                            'critic_target': self.critic_target.state_dict(),
                            'reward_decoder': self.reward_decoder.state_dict(),
                            'transition_model': self.transition_model.state_dict(),
                            'log_alpha': self.log_alpha,
                            'target_entropy': self.target_entropy,
                            'state_reward_decoder': self.state_reward_decoder.state_dict()
                            }
        
        # state_reward_decoder's params are included in decoder_optimizer
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



    def to(self, device):
        self.device = device
        self.actor = self.actor.to(device=device)
        self.critic = self.critic.to(device=device)
        self.critic_target = self.critic_target.to(device=device)
        self.log_alpha = self.log_alpha.to(device=device)
        self.state_reward_decoder = self.state_reward_decoder.to(device=device)
        return self
    
    def _sqrt(self, x, tol=None):
        # tol = torch.zeros_like(x)
        tol = tol or self.epsilon
        tol = torch.ones_like(x)*tol
        return torch.sqrt(torch.maximum(x, tol))

    def _cosine_distance(self, x, y):
        numerator = torch.sum(x * y, dim=-1, keepdim=True)
        # print("numerator", numerator.shape, numerator)
        denominator = torch.sqrt(
            torch.sum(x.pow(2.), dim=-1, keepdim=True)) * torch.sqrt(torch.sum(y.pow(2.), dim=-1, keepdim=True))
        cos_similarity = numerator / (denominator + self.epsilon)

        return torch.atan2(self._sqrt(1. - cos_similarity.pow(2.)), cos_similarity)


class StateRewardDecoder(nn.Module):
    def __init__(self, encoder_feature_dim, max_sigma=1e0, min_sigma=1e-4):
        super().__init__()
        self.trunck = nn.Sequential(
            nn.Linear(encoder_feature_dim, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Linear(512, 2))

        self.max_sigma = max_sigma
        self.min_sigma = min_sigma

    def forward(self, x):
        y = self.trunck(x)
        sigma = y[..., 1:2]
        mu = y[..., 0:1]
        sigma = torch.sigmoid(sigma)  # range (0, 1.)
        sigma = self.min_sigma + (self.max_sigma - self.min_sigma) * sigma  # scaled range (min_sigma, max_sigma)
        return mu, sigma
       
    
    def loss(self, mu, sigma, r, reduce='mean'):
        diff = (mu - r.detach()) / sigma
        if reduce == 'none':
            loss = 0.5 * (0.5 * diff.pow(2) + torch.log(sigma))
        elif  reduce =='mean':
            loss = 0.5 * torch.mean(0.5 * diff.pow(2) + torch.log(sigma))
        else:
            raise NotImplementedError

        return loss
    


class NeuralEFRAPBisimAgent(RAPBisimAgent):
    def __init__(self, *args, **kwargs):
        self.normalize_kernel = kwargs.pop('normalize_kernel', True)
        self.kernel_type = kwargs.pop('kernel_type', 'gaussian')
        super().__init__(*args, **kwargs)

    def update_encoder(self, obs, action, reward, L=None, step=None, next_obs=None):

        h = self.critic.encoder(obs)            

        # Reward predictions
        reward_mu, reward_sigma = self.state_reward_decoder(h)
        with torch.no_grad():
            pred_next_latent_mu, _ = self.transition_model(torch.cat([h, action], dim=1))
            distances = self._distance(reward, reward_sigma, pred_next_latent_mu)
            # Get the kernel/weights matrix
            kernel = self._kernel(distances, kernel_bandwidth='auto')

            if self.normalize_kernel:
                D_sqrt = torch.diag(torch.sum(kernel, dim=1)**(-0.5))
                #TODO: Make faster. Shouldn't waste time multiplying with diagonal matrix
                kernel = D_sqrt @ kernel @ D_sqrt

        loss, loss_dict = self._spectral_loss(kernel, h)
        # Add reward_decoder_loss
        loss_reward_decoder = self.state_reward_decoder.loss(
            reward_mu, reward_sigma, reward
        )
        loss = loss + loss_reward_decoder
        loss_dict['reward_decoder_loss'] = loss_reward_decoder.item()
        
        loss_dict['embedding_norm'] = torch.norm(h, dim=1).mean().item()

        return loss, loss_dict    

    def _distance(self, reward, reward_sigma, pred_next_latent_mu, next_obs=None):

        with torch.no_grad():
            r_dist = torch.cdist(reward, reward, p=1).pow(2.)

            reward_variance = reward_sigma.detach().pow(2.) # (B,1)
            r_variance_dist = reward_variance[:,:,None] + reward_variance[None,:,:] # pairwise sums of vars
            r_variance_dist = r_variance_dist.squeeze(-1)
            # TODO: Without relu, the square root of negative values will result in nans
            #       But the paper doesn't mention this relu step
            r_dist = F.relu(r_dist - r_variance_dist)
            r_dist = self._sqrt(r_dist)
            # Use L2-distance on the latent space for transition distances
            transition_dist = torch.cdist(pred_next_latent_mu, pred_next_latent_mu, p=2)
            # if self.transition_model_type in ['', 'deterministic']:
            #     # 2-norm distance between next latents
            #     transition_dist = torch.cdist(pred_next_latent_mu, pred_next_latent_mu, p=2) # shape (B,B)
            # else:
            #     transition_dist = self.metric_func(pred_next_latent_mu, pred_next_latent_mu)
 
            return r_dist + self.discount * transition_dist
    
    # def metric_func(self, x, y):
    #     # if self.rap_structural_distance == 'l2':
    #     dist = torch.cdist(x, y, p=2)
        # elif self.rap_structural_distance == 'l1':
        #     dist = torch.cdist(x, y, p=1)
        # elif self.rap_structural_distance == 'mico_angular':
        #     beta = 1e-6 #1e-5 # #0.1
        #     base_distances = self._cosine_distance(x, y)
        #     # print("base_distances", base_distances)
        #     norm_average = (x.pow(2.).sum(dim=-1, keepdim=True)
        #     # norm_average = 0.5 * (x.pow(2.).sum(dim=-1, keepdim=True) 
        #         + y.pow(2.).sum(dim=-1, keepdim=True))
        #     dist = norm_average + (beta * base_distances)
        # elif self.rap_structural_distance == 'x^2+y^2-xy':
        #     raise NotImplementedError
        #     # # beta = 1.0 # 0 < beta < 2
        #     # k = 0.1 # 0 < k < 2
        #     # base_distances = (x * y).sum(dim=-1, keepdim=True)
        #     # # print("base_distances", base_distances)
        #     # norm_average = (x.pow(2.).sum(dim=-1, keepdim=True) 
        #     #     + y.pow(2.).sum(dim=-1, keepdim=True))
        #     # # dist = norm_average - (2. - beta) * base_distances
        #     # dist = norm_average - k * base_distances
        #     # # dist = dist.sqrt()
        # else:
        #     raise NotImplementedError
        return dist


    def _kernel(self, distances, kernel_bandwidth='auto'):
        if kernel_bandwidth == 'auto':
            kernel_bandwidth = (2*(torch.median(distances)*2))
            kernel_bandwidth = kernel_bandwidth if kernel_bandwidth > 0 else 1.0
            nu = 1./kernel_bandwidth
        else:
            nu = 1./(2*(kernel_bandwidth**2))
        W = torch.exp(-nu*(distances)**2) # shape (B, B)
        return W
    

    def _spectral_loss(self, W, features):
        """
        W : kernel matrix (weights matrix)
        """
        loss_dict = {}

        batch_size = W.shape[0]

        psis_x = features
        kernel = W

        K_psis = kernel @ psis_x
        psis_K_psis = psis_x.T @ K_psis
        psisSG_K_psis = psis_x.T.clone().detach() @ K_psis

        # Keep track of the eigenvalues
        if hasattr(self.critic.encoder, 'eigvals'):
            with torch.no_grad():
                eigvals = psis_K_psis.diag()/(batch_size**2)
                if self.critic.encoder.num_eigval_calls == 0:
                    self.critic.encoder.eigvals.copy_(eigvals.data)
                    self.actor.encoder.eigvals.copy_(eigvals.data)
                else:
                    self.critic.encoder.eigvals.mul(self.critic.encoder.eigval_momentum).add_(
                        eigvals.data, alpha = 1-self.critic.encoder.eigval_momentum
                    )
                    self.actor.encoder.eigvals.mul(self.actor.encoder.eigval_momentum).add_(
                        eigvals.data, alpha = 1-self.actor.encoder.eigval_momentum
                    )
                self.critic.encoder.num_eigval_calls += 1
                self.actor.encoder.num_eigval_calls += 1
                loss_dict['eigenvalues'] = self.actor.encoder.eigvals.detach().cpu().numpy()


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


    def _cosine_distance(self, x, y):
        """
        Pair-wise cosine distances
        """
        #TODO: Replace scipy's consine similarity with torch functions
        cos_theta = pairwise_consine_similarity(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        cos_theta = torch.from_numpy(cos_theta).to(x.device)
        sin_theta = self._sqrt(1. - cos_theta.pow(2.)) # sinx = sqrt(1-cos(x)^2)
        return torch.atan2(sin_theta, cos_theta) # theta = arctan(sin/cos)