from src.models.encoder import VectorEncoder
import torch

class EigenVectorEncoder(VectorEncoder):
    
    def __init__(self, obs_shape, feature_dim, eigen_mode='spectral_net', **kwargs):
        super().__init__(obs_shape, feature_dim)
        self.eigen_mode = eigen_mode
        self.normalize_loss = kwargs.get('normalize_loss', False)


    def update_encoder(self, obs, weights):

        # Get latents
        latents = self.forward(obs)
        # Get distances
        distances = self._get_distances(obs)
        # Get kernel/weights
        K = self._get_kernel(distances)
        # Get loss
        loss = self._get_loss(latents, K)

    def _get_kernel(self, distances):
        pass
    
    def _get_distances(self, obs):
        pass

    def _get_loss(self, encodings, kernel):
        """
        """
        pass

    def _loss_spectral_net(self, latents, kernel):
        batch_size = latents.shape[0]
        if self.normalize_loss:
            D = torch.sum(kernel, dim=1)
            latents = latents / D[:, None]
        Dh = torch.cdist(latents, latents, p=2) 
        
        # distance loss
        dist_loss = torch.sum(kernel * Dh.pow(2)) / (batch_size**2)

        # orthogonality loss
        D = torch.diag(torch.sum(kernel, dim=1))
        est = (latents.T @ D @ latents)
        ortho_loss = (est - torch.eye(latents.shape[1]).to(latents.device))**2
        # ortho_loss = (((Y.T@Y)*(1/Y.shape[0]) - th.eye(Y.shape[1]).to(Y.device))**2)
        ortho_loss = ortho_loss.sum()/(latents.shape[1])
        loss = loss + self.encoder_ortho_loss_reg*ortho_loss

    def _loss_neural_efs(self, latents, kernel):
        pass