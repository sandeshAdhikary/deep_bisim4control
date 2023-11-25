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
from trainer.utils import import_module_attr
from functools import partial
from trainer.study import Study
from tqdm import trange
from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image

from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, MultiStepLR, ExponentialLR
SCHEDULERS = {
     'cosine': CosineAnnealingLR,
     'linear': LinearLR,
}

### Datasets
class GridworldDataset(Dataset):
    """
    Dataset object for training, validation, and testing and data
    """
    
    DATA_PATHS = {
        'train':'src/neural_ef_study/data/data.npy',
    }

    def __init__(self, mode='train'):
        self.data = np.load(self.DATA_PATHS[mode]).astype(np.float64)
        self.grid_size = 20
        self.bandwidth = 1.0

    def __getitem__(self, index):
        """
        First 6 elements are the data, last element is the label
        """
        return self.data[index]

    def __len__(self):
        return len(self.data)
    
    def get_kernel(self, X):
         """
         Return kernel matrix for data points in X
         """
         pass

    def precomp_kernel_laplace(self, normalized=True):

        # obstacles = np.concatenate(
        #     [
        #         np.array([(int(self.grid_size/2), x) for x in range(self.grid_size) if x not in [4,5,6,7]]),
        #         np.array([(x, int(self.grid_size/2)) for x in range(self.grid_size) if x not in [4,5,6,7]]),
        #     ]
        #     )

        obstacles = None

        self.neighbor_eps = 20
        dist_mat = np.zeros((self.grid_size, self.grid_size, self.grid_size, self.grid_size))
        for r1 in trange(self.grid_size):
            for c1 in range(self.grid_size):
                for r2 in range(self.grid_size):
                    for c2 in range(self.grid_size):

                        if r1 == r2 and c1 == c2:
                            dist_mat[r1,c1,r2,c2] = 0.0
                        else:
                            if obstacles is not None:
                                if [r1,c1] in obstacles.tolist() or [r2,c2] in obstacles.tolist():
                                    dist_mat[r1,c1,r2,c2] = np.inf
                                    continue

                            if (abs((r2 - r1)) <= self.neighbor_eps) and (abs((c2 - c1)) <= self.neighbor_eps):
                                dist_mat[r1,c1,r2,c2] = np.sqrt((r1-r2)**2 + (c1-c2)**2)

        kernel = np.exp(-(dist_mat**2)/(2*self.bandwidth**2))
        kernel = rearrange(kernel, 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)', r1=self.grid_size, c1=self.grid_size)

        # # Get the Laplacian
        D = np.diag(np.sum(kernel, axis=0))
        laplacian = (D - kernel).copy()
        
        if normalized:
            # Normalize the kernel and Laplacian
            # D_sqrt = np.diag(np.diag(kernel**(-0.5)))
            D_sqrt = np.diag(np.sum(kernel, axis=0)**(-0.5))
            kernel = D_sqrt @ kernel @ D_sqrt
            laplacian = D_sqrt @ laplacian @ D_sqrt

        
        kernel = rearrange(kernel, ' (r1 c1) (r2 c2) -> r1 c1 r2 c2', r1=self.grid_size, r2=self.grid_size)
        laplacian = rearrange(laplacian, '(r1 c1) (r2 c2) -> r1 c1 r2 c2', r1=self.grid_size, r2=self.grid_size)

        return torch.from_numpy(kernel), torch.from_numpy(laplacian)

class NeuralEFTrainer(SupervisedTrainer):
    
    def setup_data(self, config):
        self.grid_data = GridworldDataset(mode='train')
        self.train_data = DataLoader(self.grid_data, 
                                        batch_size=config['batch_size'], 
                                        shuffle=False)
        self.kernel, self.laplacian = self.grid_data.precomp_kernel_laplace()

    def _get_kernel(self, X):
        self.grid_data.get_kernel(X)
        
    def log_epoch(self, config):

        self.model.eval()

        loss = self.train_log[-1]['log']['loss']
        self.logger.log(log_dict={'loss': loss, 'trainer_step': self.epoch})

        current_lr = self.model.optimizer.param_groups[0]['lr']
        self.logger.log(log_dict={'lr': current_lr, 'trainer_step': self.epoch})


        if self.epoch == 1:
            # plot image of the target kernel
            kernel = rearrange(self.kernel.numpy(), 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
            kernel_img = self.make_heatmap(kernel)
            self.logger.log_image(f'data/kernel', torch.from_numpy(kernel_img), image_mode='chw')
             
            # Plot the laplacian
            laplacian = rearrange(self.laplacian.numpy(), 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
            laplacian_img = self.make_heatmap(laplacian)
            self.logger.log_image(f'data/laplacian', torch.from_numpy(laplacian_img), image_mode='chw')

            # Plot the Laplacian's smallest eigenvectors
            D = np.diag(np.sum(kernel, axis=0))
            # eigvals, eigvecs = torch.lobpcg(torch.from_numpy(laplacian).to('cpu'), B=torch.from_numpy(D), k=self.model.k, largest=False)
            eigvals, eigvecs = torch.lobpcg(torch.from_numpy(laplacian).to('cpu'), k=self.model.k, largest=False)
            eigvecs = rearrange(eigvecs, '(r1 c1) d -> r1 c1 d', r1=int(np.sqrt(eigvecs.shape[0])))
            max_viz_dim = min(eigvecs.shape[-1], 4)
            for idx in range(max_viz_dim):
                eigvec = self.make_heatmap(eigvecs[:,:,idx].numpy())
                self.logger.log_image(f'data/laplacian_eigvecs_{idx}', eigvec, image_mode='chw')

            # Plot the normalized kernel's largest eigenvectors
            eigvals, eigvecs = torch.lobpcg(torch.from_numpy(kernel).to('cpu'), k=self.model.k, largest=True)
            eigvecs = rearrange(eigvecs, '(r1 c1) d -> r1 c1 d', r1=int(np.sqrt(eigvecs.shape[0])))
            max_viz_dim = min(eigvecs.shape[-1], 4)
            for idx in range(max_viz_dim):
                eigvec = self.make_heatmap(eigvecs[:,:,idx].numpy())
                self.logger.log_image(f'data/kernel_eigvecs_{idx}', eigvec, image_mode='chw')

        # # Plot loss between true and estimated eigenvectors
        if self.epoch % 50 == 0:
            with torch.no_grad():
                data = self.train_data.dataset.data
                eigvecs = self.model(torch.from_numpy(data)).to('cpu').numpy()
                eigvecs = rearrange(eigvecs, '(r1 c1) d -> r1 c1 d', r1=int(np.sqrt(eigvecs.shape[0])))
                true_eigvecs = self.model.true_eigs.to('cpu').numpy()
                true_eigvecs = rearrange(true_eigvecs, '(r1 c1) d -> r1 c1 d', r1=int(np.sqrt(true_eigvecs.shape[0])))
                # get plots of the top-n eigenvectors vs encodings
                max_viz_dim = min(eigvecs.shape[-1], 6)
                for idx in range(max_viz_dim):
                    encoding = self.make_heatmap(eigvecs[:,:,idx])
                    eigvec = self.make_heatmap(true_eigvecs[:,:,idx])
                    img = make_grid([torch.from_numpy(encoding), torch.from_numpy(eigvec)], nrow=1)
                    self.logger.log_image(f'eval/encodigs_{idx}', img, image_mode='chw')

    def make_heatmap(self, array, cmap='viridis', img_mode='chw'):
        """
        """
        # Normalize the input array to be in the range [0, 1]
        normalized_array = (array - np.min(array)) / (np.max(array) - np.min(array))

        # Choose the colormap
        colormap = plt.get_cmap(cmap)

        # Map the normalized values to RGBA values using the chosen colormap
        img = (colormap(normalized_array) * 255).astype(np.uint8)

        img = Image.fromarray(img).resize((84, 84)).convert('RGB')
        img = np.array(img)
        
        if img_mode == 'chw':
            img = rearrange(img, 'h w c -> c h w')
        return img

                

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
        self.domain = config.get('domain', 'gridworld')
        self.hidden_dim = config.get('hidden_dim', 32)
        self.nonlinearity = config.get('nonlinearity', 'relu')

        # lr = config.get('lr', 1e-3)
        model = NeuralEigenFunctions(self.k, hidden_size=self.hidden_dim, nonlinearity=self.nonlinearity)
        model = model.double()
        optimizer = torch.optim.Adam(model.parameters(), 
                                     **config['optimizer']['optimizer_kwargs'])
        scheduler = None
        if config.get('scheduler', None) is not None:
            scheduler_cls = SCHEDULERS[config['scheduler']['type']]
            scheduler = scheduler_cls(optimizer, **config['scheduler']['scheduler_kwargs'])

        super().__init__(config, model, optimizer, None, scheduler=scheduler)

        self.kernel = self.kernel.to(self.device).double()

        self.true_eigs = self._get_true_eigs(self.kernel)

    def _get_true_eigs(self, kernel): 
        kernel = rearrange(kernel, 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
        _, eigvecs = torch.lobpcg(kernel.to('cpu'), k=self.k, largest=True)
        return eigvecs
    
    def _get_kernel(self, X):
        kernel = torch.zeros((X.shape[0], X.shape[0]))
        for idx in range(X.shape[0]):
                for idy in range(X.shape[0]):
                    x1, y1 = int(X[idx][0]), int(X[idx][1])
                    x2, y2 = int(X[idy][0]), int(X[idy][1])
                    kernel[idx,idy] = self.kernel[x1,y1,x2,y2]
            
        return kernel.double()

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
        
        # idx = np.random.choice(x.shape[0], batch_size, replace=False)
        # x_batch = x[idx]
        x_batch = x
        psis_x = self.forward(x_batch)
        with torch.no_grad():
            # kernel = self._get_kernel(x_batch.to('cpu')).to(psis_x.device)
            kernel = self.kernel.to(psis_x.device)
            # kernel = kernel.clip(min=1e-8)
            kernel = rearrange(kernel, 'r1 c1 r2 c2 -> (r1 c1) (r2 c2)')
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

             loss = (r_jj - r_ij)
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