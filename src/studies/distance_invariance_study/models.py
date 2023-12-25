from torch import nn
from src.NeuralEigenFunction.utils import nystrom, psd_safe_cholesky, rbf_kernel, \
	polynomial_kernel, periodic_plus_rbf_kernel, build_mlp_given_config, ParallelMLP
from src.NeuralEigenFunction.utils import SinAndCos, Erf
import math
from tqdm import trange
import torch
import numpy as np
from functools import partial
from trainer.model import Model

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

class SingleCNNEigenFunctions(nn.Module):
    def __init__(self, k, nonlinearity='sin_and_cos', input_size=1,
                 hidden_size=32, num_layers=3, output_size=1, momentum=0.9,
                 normalize_over=[0]):
         super().__init__()
         if nonlinearity == 'relu':
              nonlinearity=nn.ReLU
         elif nonlinearity == 'sin_and_cos':
              nonlinearity=SinAndCos
         else:
            raise NotImplementedError
         
         self.conv_layers = nn.Sequential(
              nn.Conv2d(3, hidden_size, kernel_size=8, stride=4),
              nonlinearity(),
              nn.Conv2d(hidden_size, hidden_size*2, kernel_size=4, stride=2),
              nonlinearity(),
              nn.Conv2d(hidden_size*2, hidden_size*2, kernel_size=3, stride=1),
              nonlinearity()
              )
         self.fc_layers = nn.Sequential(
              nn.Linear(self._calculate_fc_input_size(3), 512),
              nonlinearity(),
              nn.Linear(512, k)
              )
         self.fn = nn.Sequential(self.conv_layers, nn.Flatten(), self.fc_layers)

         self.momentum = momentum
         self.normalize_over = normalize_over
         self.register_buffer('eigennorm', torch.zeros(k))
         self.register_buffer('num_calls', torch.Tensor([0]))

    def _calculate_fc_input_size(self, input_channels):
        # Dummy input to calculate the size after convolutional layers
        dummy_input = torch.zeros((1, input_channels, 84, 84), dtype=torch.float32)
        dummy_output = self.conv_layers(dummy_input)
        return dummy_output.view(dummy_output.size(0), -1).size(1)

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

class EigenModel(Model):
    def __init__(self, config, trainer):
        self.k = config.get('encoder_dim', 10)
        self.kernel = trainer.kernel
        self._get_kernel = trainer.get_kernel
        self._get_distances = trainer.get_distances
        # self.kernel_type = config.get('kernel_type', 'rbf')
        self.domain = config.get('domain')
        self.input_size = config.get('input_size', 1)
        self.eigenvalues = None

        self.model_type = config.get('model_type', 'SingleMLPEigenFunctions')
        if self.model_type == 'MultiMLPEigenFunctions':
            model = MultiMLPEigenFunctions(self.k, input_size=self.input_size)
        elif self.model_type == 'SingleMLPEigenFunctions':
             model = SingleMLPEigenFunctions(self.k, input_size=self.input_size)
        elif self.model_type == 'SingleCNNEigenFunctions':
             model = SingleCNNEigenFunctions(self.k)
        else:
            raise ValueError(f"model_type {self.model_type} not recognized")

        # Add downstream model 
        downstream_hidden_dim = config.get('downstream_hidden_dim', 32)
        downstream_output_dim = config.get('downstream_output_dim', 1)
        downstream_model = torch.nn.Sequential(
            torch.nn.Linear(self.k, downstream_hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(downstream_hidden_dim, downstream_output_dim)
        )

        model = torch.nn.ModuleDict({'encoder': model, 'downstream': downstream_model})

        optimizer = torch.optim.Adam(model.parameters(), 
                                     **config['optimizer']['optimizer_kwargs'])
        scheduler = None
        if config.get('scheduler', None) is not None:
            if config['scheduler']['name'] == 'CosineAnnealingLR':
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                    **config['scheduler']['scheduler_kwargs'])
            elif config['scheduler']['name'] == 'LinearLR':
                scheduler = torch.optim.lr_scheduler.LinearLR(optimizer, **config['scheduler']['scheduler_kwargs'])

        super().__init__(config, model, optimizer, self.loss_fn, scheduler=scheduler)

        self.kernel = self.kernel.to(self.device)

        self.true_eigs = self._get_true_eigs(self.kernel)

    def forward(self, x):
        if x.device != self.device:
            x = x.to(self.device)
        encodings = self.model['encoder'](x)
        preds = self.model['downstream'](encodings)
        return encodings, preds

    def loss_fn(self, x, encodings, indices, preds, y=None):
        #(x, pred, y)
        return None

    def _get_true_eigs(self, kernel): 
        # Compute the true eigenfunctions
        eigvals, eigvecs = torch.linalg.eigh(kernel.to('cpu'))
        return eigvecs[:, -self.k:].flip(dims=[1])
    
    def training_step(self, batch, batch_idx):
        """
        Compute single training_step for a given batch, and update model parameters
        returns the loss
        """
        x = batch.get('data')
        y = batch.get('label')
        indices = batch.get('indices')

        encodings, preds = self.forward(x)
        self.zero_grad()
        loss = self.loss(x, encodings, indices=indices, preds=preds, y=y)
        loss.backward()
        self.optimizer.step()

        if self.scheduler is not None:
            self.scheduler.step()
    
        return {
            'x': x.detach().cpu().numpy(),
            'y': y.detach().cpu().numpy() if y is not None else y,
            'indices': indices.detach().cpu().numpy(),
            'encodings': encodings.detach().cpu().numpy(),
            'preds': preds.detach().cpu().numpy() if y is not None else y,
            'loss': loss.view(-1).detach().cpu().numpy(),
            'model': self.state_dict(include_optimizers=False)
        }


    def evaluation_step(self, batch, batch_idx):
        """
        Compute single evaluation step for a given batch
        returns: output dict with keys x, y, preds, loss
        """
        self.model.eval()

        data, labels, indices = batch['data'], batch['label'], batch['indices']

        # Get predictions; also compute gradients wrt inputs
        data.requires_grad = True
        encodings, preds = self.forward(data)
        self.model.zero_grad()
        preds.sum().backward()
        grad = data.grad.abs()
        data.requires_grad = False
        with torch.no_grad():
            #  loss_fn(self, x, encodings, indices, preds, y=None)
                loss = self.loss(data, encodings=encodings, indices=indices, preds=preds, y=labels)

        return {'x': data.detach().cpu().numpy(),
                'y': labels.detach().cpu().numpy(),
                'indices': indices.detach().cpu().numpy(),
                'encodings': encodings.detach().cpu().numpy(),
                'preds': preds.detach().cpu().numpy() if labels is not None else None,
                'grads': grad.detach().cpu().numpy() if labels is not None else None,
                'loss': loss.view(-1).detach().cpu().numpy(),
                }


class BaselineNet(EigenModel):

    def loss_fn(self, x, encodings, indices=None, preds=None, y=None):
        # Just prediction loss
        assert (preds is not None) and (y is not None)
        loss = torch.nn.functional.mse_loss(preds.to(y.dtype), y)
        return loss

class ApproxSpectralNet(EigenModel):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.normalize_loss = self.config.get('normalize_loss', True)
        self.regularization = self.config.get('regularization', 1.0)

    def loss_fn(self, x, encodings, indices=None, preds=None, y=None):

        with torch.no_grad():
            kernel = self._get_kernel(x.to('cpu')).to(encodings.device)

        batch_size = kernel.shape[0]

        if self.normalize_loss:
            D = torch.sum(kernel, dim=1)
            encodings = encodings / D[:, None]

        Dh = torch.cdist(encodings, encodings, p=2)
        loss = torch.sum(kernel * Dh.pow(2)) / (batch_size**2)

        # Add orthogonality loss
        D = torch.diag(torch.sum(kernel, dim=1))
        est = (encodings.T @ D @ encodings)
        ortho_loss = (est - torch.eye(encodings.shape[1]).to(encodings.device))**2
        ortho_loss = ortho_loss.sum()/(encodings.shape[1])
        loss = loss + self.regularization*ortho_loss


        # If target provided, include prediction loss as well
        if (y is not None) and (preds is not None):
            loss += torch.nn.functional.mse_loss(preds.to(y.dtype), y)

        return loss

class DistanceMatchModel(EigenModel):
    
    def loss_fn(self, x, encodings, indices, preds, y):

        # Get distances
        with torch.no_grad():
            distances = self._get_distances(indices.to('cpu')).to(encodings.device)
        
        # Compute L1-distance in the latent space of predictions
        latent_dists = torch.cdist(encodings, encodings, p=1)

        # Loss is the difference between pred_dists and actual distances
        loss = torch.nn.functional.l1_loss(latent_dists, distances)
        batch_size = encodings.shape[0]
        loss /= batch_size**2
        
        # If target provided, include prediction loss as well
        if (y is not None) and (preds is not None):
            loss += torch.nn.functional.mse_loss(preds.to(y.dtype), y)

        return loss


class NeuralEFModel(EigenModel):

    # def training_step(self, batch, batch_idx):
    #     """
    #     Overwriting default training_step since NeuralEF requires a custom backward pass
    #     where some of the gradient is computed manually
    #     """
    #     self.model.train()

    #     x = batch.get('data')
    #     y = batch.get('label')
    #     indices = batch.get('indices')

    #     batch_size = x.shape[0]
        
    #     self.zero_grad()        
    #     psis_x, preds = self.forward(x)
    #     with torch.no_grad():
    #         kernel = self._get_kernel(indices.to('cpu')).to(psis_x.device)
    #         K_psis = kernel @ psis_x 
    #         psis_K_psis = psis_x.T @ K_psis
    #         mask = torch.eye(self.k, device=psis_x.device) - \
    #             (psis_K_psis / psis_K_psis.diag()).tril(diagonal=-1).T
    #         grad = K_psis @ mask
    #         if self.eigenvalues is None:
    #             self.eigenvalues = psis_K_psis.diag() / (batch_size**2)
    #         else:
    #             self.eigenvalues.mul_(0.9).add_(psis_K_psis.diag() / (batch_size**2), alpha = 0.1)
    #         grad = grad/torch.norm(grad)
        
    #     psis_x.backward(-grad)
    #     self.optimizer.step()
                
       
    #     loss = torch.tensor(0.0)
    #     if y is not None:
    #         self.zero_grad()
    #         psis_x, preds = self.forward(x)
    #         loss = torch.nn.functional.mse_loss(preds.to(y.dtype), y.to(preds.device))
    #         loss.backward()
    #         self.optimizer.step()
            

    #     with torch.no_grad():
    #          r_jj = torch.trace(psis_K_psis)
    #          r_ij = torch.triu(psis_K_psis**2, diagonal=1).sum(dim=1)
    #          r_ij = r_ij.sum()
    #          loss += -(r_jj - r_ij)
    
    #     if self.scheduler is not None:
    #         self.scheduler.step()    

    #     return {
    #         'x': x.detach().cpu().numpy(),
    #         'y': y.detach().cpu().numpy() if y is not None else y,
    #         'indices': indices.detach().cpu().numpy(),
    #         'encodings': psis_x.detach().cpu().numpy(),
    #         'preds': preds.detach().cpu().numpy() if y is not None else y,
    #         'loss': loss.view(-1).detach().cpu().numpy(),
    #         'model': self.state_dict(include_optimizers=False)
    #     }


    # def loss_fn(self, x, encodings, indices, preds, y=None):
        
    #     psis_x = encodings
    #     with torch.no_grad():
    #         kernel = self._get_kernel(indices.to('cpu')).to(psis_x.device)
    #         K_psis = kernel @ psis_x 
    #         psis_K_psis = psis_x.T @ K_psis
        
    #     loss = torch.tensor(0.0)
    #     if y is not None:
    #         loss = torch.nn.functional.mse_loss(preds, y.to(preds.device))
            
    #     with torch.no_grad():
    #          r_jj = torch.trace(psis_K_psis)
    #          r_ij = torch.triu(psis_K_psis**2, diagonal=1).sum(dim=1)
    #          r_ij = r_ij.sum()
    #          loss += -(r_jj - r_ij)

    #     return loss

    def loss_fn(self, x, encodings, indices, preds, y=None):

        batch_size = x.shape[0]
        
        self.zero_grad()        

        psis_x, preds = self.forward(x)
        with torch.no_grad():
            kernel = self._get_kernel(indices.to('cpu')).to(psis_x.device)
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

        if y is not None:
            loss += torch.nn.functional.mse_loss(preds.to(y.dtype), y.to(preds.device))

        return loss