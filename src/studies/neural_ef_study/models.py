from torch import nn
from src.neural_ef_study.NeuralEigenFunction.utils import nystrom, psd_safe_cholesky, rbf_kernel, \
	polynomial_kernel, periodic_plus_rbf_kernel, build_mlp_given_config, ParallelMLP
import math
from tqdm import trange
import torch
import numpy as np
from functools import partial
from trainer.model import EncoderModel

class MultiMLPEigenFunctions(nn.Module):
	def __init__(self, k, nonlinearity='sin_and_cos', input_size=1,
				 hidden_size=32, num_layers=3, output_size=1, momentum=0.9,
				 normalize_over=[0]):
		super(MultiMLPEigenFunctions, self).__init__()
		self.momentum = momentum
		self.normalize_over = normalize_over
		self.fn = ParallelMLP(input_size, output_size, k, num_layers, hidden_size, nonlinearity)
		self.register_buffer('eigennorm', torch.zeros(k))
		self.register_buffer('num_calls', torch.Tensor([0]))

	def forward(self, x):
		ret_raw = self.fn(x).squeeze()
		if self.training:
			norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(
				np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
			with torch.no_grad():
				if self.num_calls == 0:
					self.eigennorm.copy_(norm_.data)
				else:
					self.eigennorm.mul_(self.momentum).add_(
						norm_.data, alpha = 1-self.momentum)
				self.num_calls += 1
		else:
			norm_ = self.eigennorm
		return ret_raw / norm_

          
class SingleMLPEigenFunctions(nn.Module):
    def __init__(self, k, nonlinearity='sin_and_cos', input_size=1,
              hidden_size=32, num_layers=3, output_size=1, momentum=0.9,
              normalize_over=[0]):
          super().__init__()
          from src.neural_ef_study.NeuralEigenFunction.utils import SinAndCos, Erf
          if nonlinearity == 'relu':
               nonlinearity=nn.ReLU
          elif nonlinearity == 'sin_and_cos':
               nonlinearity=SinAndCos
          else:
            raise NotImplementedError

          self.fn = nn.Sequential(
               nn.Linear(input_size, hidden_size),
               nonlinearity(),
               nn.Linear(hidden_size, hidden_size),
               nonlinearity(),
               nn.Linear(hidden_size, hidden_size),
               nonlinearity(),
               nn.Linear(hidden_size, k)
          )
          self.momentum = momentum
          self.normalize_over = normalize_over
          self.register_buffer('eigennorm', torch.zeros(k))
          self.register_buffer('num_calls', torch.Tensor([0]))
    
    def forward(self, x):
        ret_raw = self.fn(x).squeeze()
        if self.training:
            norm_ = ret_raw.norm(dim=self.normalize_over) / math.sqrt(
                 np.prod([ret_raw.shape[dim] for dim in self.normalize_over]))
            with torch.no_grad():
                if self.num_calls == 0:
                    self.eigennorm.copy_(norm_.data)
                else:
                    self.eigennorm.mul_(self.momentum).add_(norm_.data, alpha = 1-self.momentum)
                    self.num_calls += 1
        else:
            norm_ = self.eigennorm
        
        return ret_raw / norm_

class EigenModel(EncoderModel):
    def __init__(self, config, kernel):
        self.k = config.get('encoder_dim', 10)
        self.kernel = kernel
        self.kernel_type = config.get('kernel_type', 'rbf')
        self.domain = config.get('domain')
        self.input_size = config.get('input_size', 1)
        self.eigenvalues = None

        self.model_type = config.get('model_type', 'neural_ef')
        if self.model_type == 'MultiMLPEigenFunctions':
            model = MultiMLPEigenFunctions(self.k, input_size=self.input_size)
        elif self.model_type == 'SingleMLPEigenFunctions':
             model = SingleMLPEigenFunctions(self.k, input_size=self.input_size)
        else:
            raise ValueError(f"model_type {self.model_type} not recognized")

        model = model.double()
        optimizer = torch.optim.Adam(model.parameters(), 
                                     **config['optimizer']['optimizer_kwargs'])
        scheduler = None
        if config.get('scheduler', None) is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                **config['scheduler']['scheduler_kwargs'])

        super().__init__(config, model, optimizer, self.loss_fn, scheduler=scheduler)

        self.kernel = self.kernel.to(self.device).double()

        self.true_eigs = self._get_true_eigs(self.kernel)

    def loss_fn(self, x, pred, y):
        #(x, pred, y)
        return None

    def _get_true_eigs(self, kernel): 
        # Compute the true eigenfunctions
        eigvals, eigvecs = torch.linalg.eigh(kernel.to('cpu'))
        return eigvecs[:, -self.k:].flip(dims=[1])
    
    def _get_kernel(self, X):
        if self.kernel_type == 'rbf':
            kernel_fn = partial(rbf_kernel, 1, 1)
            # ylim = [-1.8, 1.5]
        elif self.kernel_type == 'polynomial':
            raise NotImplementedError
        return kernel_fn(X)

    def evaluation_step(self, batch, batch_idx):
            """
            Compute single evaluation step for a given batch
            returns: output dict with keys x, y, preds, loss
            """
            return {}


class ApproxSpectralNet(EigenModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_loss = self.config.get('normalize_loss', True)
        self.regularization = self.config.get('regularization', 100.0)

    def loss_fn(self, x, pred, y):

        with torch.no_grad():
            kernel = self._get_kernel(x.to('cpu')).to(pred.device)

        batch_size = kernel.shape[0]

        if self.normalize_loss:
            D = torch.sum(kernel, dim=1)
            pred = pred / D[:, None]

        Dh = torch.cdist(pred, pred, p=2)
        loss = torch.sum(kernel * Dh.pow(2)) / (batch_size**2)

        # Add orthogonality loss
        D = torch.diag(torch.sum(kernel, dim=1))
        est = (pred.T @ D @ pred)
        ortho_loss = (est - torch.eye(pred.shape[1]).to(pred.device))**2
        ortho_loss = ortho_loss.sum()/(pred.shape[1])
        loss = loss + self.regularization*ortho_loss

        return loss

class DistanceMatchModel(EigenModel):
    
    def loss_fn(self, x, pred, y):

        # Get distances
        with torch.no_grad():
            kernel = self._get_kernel(x.to('cpu')).to(pred.device)
            # TODO: Assuming RBF kernel. Otherwise model needs access to kernel type, and kernel params
            distances = self.rbf_distance(kernel, length_scale=1.0, output_scale=1.0)
        
        # Compute L1-distance in the latent space of predictions
        pred_dists = torch.cdist(pred, pred, p=1)

        # Loss is the difference between pred_dists and actual distances
        loss = ((distances - pred_dists)**2).mean()
        # loss = torch.linalg.norm(distances - pred_dists, ord=2)
        return loss
        
    def rbf_distance(self, kernel, length_scale=1.0, output_scale=1.0 ):
        """Given an rbf kernel matrix, compute the distance matrix"""
        return -torch.log((kernel / output_scale)) * length_scale

class NeuralEFModel(EigenModel):

    def training_step(self, batch, batch_idx):
        """
        Overwriting default training_step since NeuralEF requires a custom backward pass
        where some of the gradient is computed manually
        """
        self.model.train()

        if len(batch) == 2:
            # Target provided
            x, y = batch
        else:
            # Targett may not be provided
            x = batch
            y = None

        batch_size = x.shape[0]
        
        idx = np.random.choice(x.shape[0], batch_size, replace=False)
        x_batch = x[idx]
        # x_batch = torch.stack([x_batch, x_batch], dim=1).view(x_batch.shape[0], -1)
        psis_x = self.forward(x_batch)
        with torch.no_grad():
            kernel = self._get_kernel(x_batch.to('cpu')).to(psis_x.device)
            # kernel = self.kernel[idx][:, idx]
            K_psis = kernel @ psis_x 
            psis_K_psis = psis_x.T @ K_psis
            mask = torch.eye(self.k, device=psis_x.device) - \
                (psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T
            grad = K_psis @ mask
            if self.eigenvalues is None:
                self.eigenvalues = psis_K_psis.diag() / (batch_size**2)
            else:
                self.eigenvalues.mul_(0.9).add_(psis_K_psis.diag() / (batch_size**2), alpha = 0.1)

        self.zero_grad()
        psis_x.backward(-grad)
        self.optimizer.step()

        with torch.no_grad():
             r_jj = torch.trace(psis_K_psis)
             r_ij = torch.triu(psis_K_psis**2, diagonal=1).sum(dim=1)
             r_ij = r_ij.sum()

             loss = r_jj - r_ij
             loss = -loss # since we are maximizing
    
        if self.scheduler is not None:
            self.scheduler.step()    

        return {
            'x': x.detach().cpu().numpy(),
            'y': y.detach().cpu().numpy() if y is not None else y,
            'preds': psis_x.detach().cpu().numpy(),
            'loss': loss.view(-1).detach().cpu().numpy(),
            'model': self.state_dict(include_optimizers=False)
        }

