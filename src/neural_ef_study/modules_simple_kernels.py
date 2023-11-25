from torch.utils.data import Dataset, DataLoader
from trainer.trainer import SupervisedTrainer
from trainer.model import EncoderModel
from trainer.evaluator import SupervisedEvaluator
import torch
from torch import nn
import numpy as np
from src.neural_ef_study.NeuralEigenFunction.utils import nystrom, psd_safe_cholesky, rbf_kernel, \
	polynomial_kernel, periodic_plus_rbf_kernel, build_mlp_given_config, ParallelMLP
import math
from functools import partial
from trainer.study import Study
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image



### Datasets
class modules_simple_kernelsDataset(Dataset):
    """
    Dataset object for training, validation, and testing and data
    """
    
    DATA_PATHS = {
        'train':'src/neural_ef_study/data/data.npy',
    }

    def __init__(self, mode='train'):
        self.data = np.load(self.DATA_PATHS[mode]).astype(np.float64)
        self.grid_size = 20
        self.bandwidth = 1e-3

    def __getitem__(self, index):
        """
        First 6 elements are the data, last element is the label
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)


class NeuralEFTrainer(SupervisedTrainer):
    
    def setup_data(self, config):
        x_range = config.get('x_range', [-2,2])
        x_dim = config.get('x_dim', 1) # data dimension
        num_samples = config.get('num_samples', 256) # number of samples
        batch_size = config.get('batch_size', 256)

        X  = torch.empty(num_samples, x_dim).uniform_(x_range[0], x_range[1])
        X = X.double()
        
        self.train_data = DataLoader(X, batch_size=batch_size, shuffle=False)
        self.kernel_type = config.get('kernel_type', 'rbf')
        self.kernel = self._get_kernel(X)

    def _get_kernel(self, X):
        if self.kernel_type == 'rbf':
            kernel_fn = partial(rbf_kernel, 1, 1)
            # ylim = [-1.8, 1.5]
        elif self.kernel_type == 'polynomial':
            raise NotImplementedError
        return kernel_fn(X)

        
    def log_epoch(self, config):

        loss = self.train_log[-1]['log']['loss']
        self.logger.log(log_dict={'loss': loss, 'trainer_step': self.epoch})

        current_lr = self.model.optimizer.param_groups[0]['lr']
        self.logger.log(log_dict={'lr': current_lr, 'trainer_step': self.epoch})

        # Plot loss between true and estimated eigenvectors
        if self.epoch % 50 == 0:
            with torch.no_grad():
                eigvecs = self.model(self.train_data.dataset).to('cpu')
                eigvecs = eigvecs / eigvecs.norm(p=2, dim=0)
                true_eigvecs = self.model.true_eigs.to('cpu')
                eigvec_diff = torch.norm(eigvecs - true_eigvecs)
                self.logger.log(log_dict={'train/eigvec_diff': eigvec_diff, 'trainer_step': self.epoch})
                if self.epoch % 500 == 0:
                    # One plot for every eigenvector
                    for ide in range(5):
                        eigvec = eigvecs[:, ide]
                        true_eigvec = true_eigvecs[:, ide]
                        data = {
                            'x': [self.train_data.dataset]*2,
                            'y': [eigvec, true_eigvec],
                            'keys': ["Estimated", "True"],
                            'title': f"Eigenfunction {ide} [Epoch {self.epoch}]"
                        }
                        self.logger.log_linechart(key=f'train/eigvec_{ide}', data=data)
               

class NeuralEFEvaluator(SupervisedEvaluator):

    def setup_data(self, config):
        x_range = config.get('x_range', [-2,2])
        batch_size = config.get('batch_size', 256)

        X_val = torch.arange(x_range[0], x_range[1],
                             (x_range[1] - x_range[0]) / 2000.).view(-1, 1)
        X_val = X_val.double()
    
        self.dataloader = DataLoader(X_val, batch_size=batch_size, shuffle=False)



class NeuralEigenFunctions(nn.Module):
	def __init__(self, k, nonlinearity='sin_and_cos', input_size=1,
				 hidden_size=32, num_layers=3, output_size=1, momentum=0.9,
				 normalize_over=[0]):
		super(NeuralEigenFunctions, self).__init__()
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


class NeuralEFModel(EncoderModel):

    def __init__(self, config, kernel):
        self.k = config.get('encoder_dim', 10)
        self.eigenvalues = None
        self.kernel = kernel
        self.kernel_type = config.get('kernel_type', 'rbf')
        self.domain = config.get('domain')
        # TODO: Setup self.kernel

        # lr = config.get('lr', 1e-3)
        model = NeuralEigenFunctions(self.k)
        model = model.double()
        optimizer = torch.optim.Adam(model.parameters(), 
                                     **config['optimizer']['optimizer_kwargs'])
        scheduler = None
        if config.get('scheduler', None) is not None:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, 
                                                                **config['scheduler']['scheduler_kwargs'])

        super().__init__(config, model, optimizer, None, scheduler=scheduler)

        self.kernel = self.kernel.to(self.device).double()

        self.true_eigs = self._get_true_eigs(self.kernel)

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

    def evaluation_step(self, batch, batch_idx):
            """
            Compute single evaluation step for a given batch
            returns: output dict with keys x, y, preds, loss
            """

            if len(batch) == 2:
                # Target provided
                x, y = batch
            else:
                # Targett may not be provided
                x = batch
                y = None

            pred = self.forward(x)
            loss = torch.tensor(0.0)
            # loss = self.loss(x, pred, y)
            return {
                'x': x.detach().cpu().numpy(),
                'y': y.detach().cpu().numpy() if y is not None else y,
                'preds': pred.detach().cpu().numpy(),
                'loss': loss.view(-1).detach().cpu().numpy()
            }