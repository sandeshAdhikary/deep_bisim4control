from src.agent.bisim_agent_baseline import BisimAgent
import torch

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
        **kwargs
    ):
        self.encoder_kernel_bandwidth = encoder_kernel_bandwidth
        self.encoder_normalize_loss = encoder_normalize_loss
        self.encoder_ortho_loss_reg = encoder_ortho_loss_reg
        super().__init__(obs_shape,
                         action_shape,
                         device,
                         transition_model_type,
                         **kwargs
                         )

    def update_encoder(self, obs, action, reward, L=None, step=None):
        # Current latent
        h = self.critic.encoder(obs)
        # Predict the next latent
        with torch.no_grad():
            # Get distances
            pred_next_latent_mu1, pred_next_latent_sigma1 = self.transition_model(torch.cat([h, action], dim=1))
            distances = self._distance(reward, pred_next_latent_mu1, pred_next_latent_sigma1)
            # Get kernel/weights matrix
            W = self._weights(distances, kernel_bandwidth=self.encoder_kernel_bandwidth)
        loss, loss_dict = self._spectral_loss(W, h)
        
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

    def _distance(self, reward, pred_next_latent_mu, pred_next_latent_sigma=None):
        # Get bi-sim distances
        with torch.no_grad():
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
            bisimilarity = r_dist + self.discount*transition_dist # shape (B,B)
        return bisimilarity
    
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