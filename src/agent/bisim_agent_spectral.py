from src.agent.bisim_agent_baseline import BisimAgent
import torch
from sklearn.metrics.pairwise import cosine_similarity as pairwise_consine_similarity

class SpectralBisimAgent(BisimAgent):
    """BisimAgent but with spectral encoder"""

    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        transition_model_type,
        encoder_kernel_bandwidth='auto',
        encoder_normalize_loss=True,
        encoder_ortho_loss_reg=1e-3,
        normalize_kernel=False,
        **kwargs
    ):
        self.encoder_kernel_bandwidth = encoder_kernel_bandwidth
        self.encoder_normalize_loss = encoder_normalize_loss
        self.encoder_ortho_loss_reg = encoder_ortho_loss_reg
        self.normalize_kernel = normalize_kernel
        super().__init__(obs_shape,
                         action_shape,
                         device,
                         transition_model_type,
                         **kwargs
                         )


    def update_encoder(self, obs, action, reward, L=None, step=None, next_obs=None):
        # Current latent
        h = self.critic.encoder(obs)
        # Predict the next latent
        with torch.no_grad():
            # Get distances
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            distances = self._distance(reward, pred_next_latent_mu1, pred_next_latent_sigma1, next_obs=next_obs)
            # Get kernel/weights matrix
            W = self._weights(distances, kernel_bandwidth=self.encoder_kernel_bandwidth)

            if self.normalize_kernel:
                D_sqrt = torch.diag(torch.sum(W, dim=1)**(-0.5))
                #TODO: Make faster. Shouldn't waste time multiplying with diagonal matrix
                W = D_sqrt @ W @ D_sqrt

        loss, loss_dict = self._spectral_loss(W, h)
        
        loss_dict['embedding_norm'] = torch.norm(h, dim=1).mean().item()

        return loss, loss_dict    
        

    def _weights(self, distances, kernel_bandwidth='auto'):
        if kernel_bandwidth == 'auto':
            kernel_bandwidth = (2*(torch.median(distances)*2))
            kernel_bandwidth = kernel_bandwidth if kernel_bandwidth > 0 else 1.0
            nu = 1./kernel_bandwidth
        else:
            nu = 1./(2*(kernel_bandwidth**2))
        W = torch.exp(-nu*(distances)**2) # shape (B, B)
        return W

    def _distance(self, reward, pred_next_latent_mu, pred_next_latent_sigma=None, next_obs=None):
        with torch.no_grad():
            # no_grad since we don't backprop through distance calculations
            # the kernel is assumed given, and neural_ef's are learned wrt to the kernel
            if self.distance_type == 'bisim':
                # Get bi-sim distances
                r_dist = torch.cdist(reward, reward, p=1) # shape (B,B)
                if self.transition_model_type in ['', 'deterministic']:
                    # 2-norm distance between next latents
                    transition_dist = torch.cdist(pred_next_latent_mu, pred_next_latent_mu, p=2) # shape (B,B)
                else:
                    # Wasserstein distance
                    assert pred_next_latent_sigma is not None
                    mu_dist = torch.cdist(pred_next_latent_mu, pred_next_latent_mu, p=2) # shape (B,B)
                    sigma_sqrt = torch.sqrt(pred_next_latent_sigma)
                    sigma_dist = torch.cdist(sigma_sqrt, sigma_sqrt, p=2)
                    transition_dist = torch.sqrt(mu_dist**2 + sigma_dist**2)
                # Full bisimilarity distance
                distance = r_dist + self.discount*transition_dist # shape (B,B)
            elif self.distance_type == 'mico':
                r_dist = torch.cdist(reward, reward, p=1) # shape (B,B)
                # Note: MICO uses the frozen encoder to get these
                h_next = self.critic_target.encoder(next_obs)
                self_dists = h_next.pow(2).mean(dim=-1, keepdim=True)
                self_dists = 0.5*(self_dists[:,None] + self_dists[None,:]).squeeze(-1) # pair-wise sum of self_dists
                angular_dists = self._cosine_distance(h_next, h_next)
                distance = r_dist + self.discount*(self_dists + self.mico_beta*angular_dists)
            else:
                raise ValueError(f'Distance type {self.distance_type} not recognized')
                    
        return distance
    
    def _cosine_distance(self, x, y, epsilon=1e-8):
        #TODO: Replace scipy's consine similarity with torch functions
        cos_theta = pairwise_consine_similarity(x.detach().cpu().numpy(), y.detach().cpu().numpy())
        cos_theta = torch.from_numpy(cos_theta).to(x.device)
        tolerance = torch.ones_like(cos_theta)*epsilon# tolerance; to prevent sqrt(0)
        sin_theta = torch.sqrt(torch.maximum(1 - cos_theta.pow(2), tolerance))
        return torch.atan2(sin_theta, cos_theta) # theta = arctan(sin/cos)

    def _spectral_loss(self, W, h):
        loss_dict = {}
        batch_size = W.shape[0]

        if self.encoder_normalize_loss:
            D = torch.sum(W, dim=1)
            h = h / D[:, None]

        Dh = torch.cdist(h, h, p=2)
        loss = torch.sum(W * Dh.pow(2)) / (batch_size**2)
        loss_dict['dist_loss'] = loss.item()

        # Add orthogonality loss
        if self.encoder_ortho_loss_reg > 0:    
            D = torch.diag(torch.sum(W, dim=1))
            est = (h.T @ D @ h)
            ortho_loss = (est - torch.eye(h.shape[1]).to(h.device))**2
            ortho_loss = ortho_loss.sum()/(h.shape[1])
            loss = loss + self.encoder_ortho_loss_reg*ortho_loss
            loss_dict['ortho_loss'] = ortho_loss.item()

        loss_dict['encoder_loss'] = loss.item()

        return loss, loss_dict


class NeuralEFBisimAgent(SpectralBisimAgent):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


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