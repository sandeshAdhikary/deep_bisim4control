from torch.utils.data import Dataset, DataLoader
from trainer.trainer import SupervisedTrainer
from trainer.evaluator import SupervisedEvaluator
import torch
import numpy as np
from einops import rearrange
from torchvision.utils import make_grid
import matplotlib.pyplot as plt
from PIL import Image
from einops import rearrange
from torch.utils.data import Dataset, DataLoader
import networkx as nx
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

class InvarianceDataset(Dataset):
    def __init__(self, seed=123, gridsize=10):
        self.gridsize = gridsize
        self.seed = seed
        self.kernel_type = 'rbf'
        self.data, self.indices, self.labels = self.make_data(seed)

    def make_data(self, seed=123):
        """
        data = (x,y,z) where (x,y) are points along a 2D grid and z is random noise
        labels = f(x,y) + g(z)
        """
        # 2D grid of points
        X, Y = torch.meshgrid(torch.linspace(-1, 1, self.gridsize),
                              torch.linspace(-1, 1, self.gridsize), 
                              indexing='xy')
        # Random noise value assigned to each grid point
        Z = torch.rand(self.gridsize, self.gridsize)
        
        # data = (x,y,z) tuples
        data = torch.stack((X, Y, Z), dim=-1)

        # labels: f(x,y) + g(z)
        # labels = torch.sin(X) + torch.cos(Y) + 0.1*Z
        labels = torch.sin(X) + torch.cos(Y) + Z
        
        return data.view(-1,3), data.view(-1,3)[:,:2], labels.view(-1)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index): 
        return {'data':torch.Tensor(self.data[index]), 
                'label':torch.Tensor([self.labels[index]]),
                'indices':torch.Tensor(self.data[index,:2])
        }

    def get_kernel(self, X):
        return self.rbf_kernel(output_scale=1., length_scale=1., x1=X) # Only use first 2 dims of X for 

    def rbf_kernel(self, output_scale, length_scale, x1, x2=None):
        distances = self.get_distances(x1,x2)
        kernel = (- distances/ length_scale).exp() * output_scale
        # TODO: Apply normalization?
        return kernel


    def get_distances(self, x1, x2=None):
        if x2 is None:
            x2 = x1

        if x1.dim() == 1:
            x1 = x1.unsqueeze(-1)
        if x2.dim() == 1:
            x2 = x2.unsqueeze(-1)

        x1 = x1[:, :2] # Only use first two dimension when computing distances
        x2 = x2[:, :2] # Only use first two dimension when computing distances

        x1 = x1.flatten(1)
        x2 = x2.flatten(1)

        distances = ((x1 ** 2).sum(-1).view(-1, 1) + (x2 ** 2).sum(-1).view(1, -1) - 2 * x1 @ x2.T) / 2.
        return distances


class GraphInvarianceDataset(InvarianceDataset):
    def make_data(self, seed=123):
        
        # make graph
        self.G = nx.grid_2d_graph(self.gridsize, self.gridsize)

        horizontal_barrier = [(int(self.gridsize/2), x) for x in range(self.gridsize) if \
                              abs(x) not in [int(self.gridsize/4), int(3*self.gridsize/4)]]
        vertical_barrier = [(x, int(self.gridsize/2)) for x in range(self.gridsize) if \
                            abs(x) not in [int(self.gridsize/4), int(3*self.gridsize/4)]
                            ]
        nodes_to_disconnect = [*horizontal_barrier, *vertical_barrier]

        for node in nodes_to_disconnect:
            neighbors = list(self.G.neighbors(node))
            [self.G.remove_edge(node, x) for x in neighbors]

        # Pre-compute distances
        self.node_list = list(self.G.nodes._nodes.keys())
        self.node_list = {node:x for (node,x) in zip(self.node_list, range(len(self.node_list)))}
        self.all_dists = self._precompute_all_dists()
        
        XY = np.array(list(self.G.nodes._nodes.keys()))
        XY = torch.from_numpy(XY)
        XY = rearrange(XY, '(h w) d -> h w d', h=self.gridsize)
        # Random noise value assigned to each grid point
        Z = torch.rand(self.gridsize, self.gridsize).unsqueeze(-1)
        
        # data = (x,y,z) tuples
        data = torch.concat((XY, Z), dim=-1)

        # labels: f(x,y) + g(z)
        labels = torch.sin(XY[:,:,0]).unsqueeze(-1) + torch.cos(XY[:,:,1]).unsqueeze(-1) + 0.1*Z
        
        return data.view(-1,3), data[:,:2].reshape(-1,2), labels.view(-1)

    def _precompute_all_dists(self):
        dist_matrix = nx.floyd_warshall_numpy(self.G, nodelist=self.G.nodes, weight=None)
        dist_matrix[np.isinf(dist_matrix)] = 10e10 # replace inf with large number
        return dist_matrix

    def get_distances(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        
        x1_nodes = x1[:,:2].long()
        x2_nodes = x2[:,:2].long()
        
        dists = torch.zeros((x1.shape[0], x2.shape[0]))
        for idx in range(x1.shape[0]):
            for idy in range(x2.shape[0]):
                x1_node = self.node_list[(x1_nodes[idx][0].item(), x1_nodes[idx][1].item())]
                x2_node = self.node_list[(x2_nodes[idy][0].item(), x2_nodes[idy][1].item())]
                dists[idx, idy] = self.all_dists[x1_node, x2_node]
        return dists


class ImageInvarianceDataset(GraphInvarianceDataset):

    def __init__(self, seed=123, gridsize=10):
        self.gridsize = gridsize
        self.seed = seed
        self.kernel_type = 'rbf'
        self.pad_size = 5
        self.noise_level = 1.0
        self.base_img_size = 84
        self.data, self.indices, self.labels = self.make_data(seed)

    def make_data(self, seed=123):
        rng = torch.Generator()
        rng.manual_seed(seed)

        # set up the graph
        super().make_data()  
        
        img_data = []
        pos = []
        for idx in range(self.gridsize):
            for idy in range(self.gridsize):
                img = self._gen_image(self.gridsize, idx, idy, img_size=self.base_img_size-2*self.pad_size)
                img = torch.from_numpy(img)
                img = rearrange(img, 'h w c -> c h w')
                img_data.append(img)
                pos.append([idx,idy])
        img_data = torch.stack(img_data, dim=0)
        img_data = img_data.to(torch.float32)
        pos = torch.from_numpy(np.stack(pos))

        # Pad the image tensor with random values
        img_data = torch.nn.functional.pad(img_data, (self.pad_size, self.pad_size, self.pad_size, self.pad_size), 
                                                      mode='constant', 
                                                      value=0)
        img_data[:,:,:self.pad_size,:] += torch.rand(img_data[:,:,:self.pad_size,:].shape, generator=rng)*255
        img_data[:,:,:,:self.pad_size] += torch.rand(img_data[:,:,:,:self.pad_size].shape, generator=rng)*255
        img_data[:,:,-self.pad_size:,:] += torch.rand(img_data[:,:,-self.pad_size:,:].shape, generator=rng)*255
        img_data[:,:,:,-self.pad_size:] += torch.rand(img_data[:,:,:,-self.pad_size:].shape, generator=rng)*255


        # Get labels that depend on positions
        # labels = torch.sin(pos[:,0]) + torch.cos(pos[:,1])
        labels = torch.sin((4*torch.pi)*pos[:,0]/pos[:,0].max())
        labels += torch.cos((4*torch.pi)*pos[:,1]/pos[:,1].max())
        # labels = torch.ones_like(pos[:,0]).to(torch.float32)
        # add dependency on the random padding
        noise = img_data[:,:,:self.pad_size,:].sum(dim=(1,2,3)) + \
            img_data[:,:,:,:self.pad_size].sum(dim=(1,2,3))+ \
            img_data[:,:,-self.pad_size:,:].sum(dim=(1,2,3)) + \
                img_data[:,:,:,-self.pad_size:].sum(dim=(1,2,3))
        noise = (noise)/(self.pad_size*(self.gridsize**2)*4*255) # keep in range 0-1
        labels += self.noise_level*noise

        return img_data.to(torch.float32), pos, labels

    def _gen_image(self, grid_size, x, y, img_size=84):
        cell_size = 4
        grid_size *= cell_size
        

        grid = np.zeros((grid_size+1, grid_size+1, 3), dtype=np.float32)
        # Color the cell at (x, y) green
        grid[y * cell_size:(y + 1) * cell_size, x * cell_size:(x + 1) * cell_size, 1] = 1.0 
        # Normalize values to [0, 1] for better visualization
        grid = grid / np.max(grid)
        # Create a Pillow image from the grid
        grid_image = Image.fromarray((grid * 255).astype(np.uint8), mode='RGB')

        # Draw gridlines
        draw = ImageDraw.Draw(grid_image)
        for i in range(0, cell_size*grid_size, cell_size):
            draw.line([(i, 0), (i, cell_size*grid_size)], fill=(128, 128, 128), width=1)
            draw.line([(0, i), (cell_size*grid_size, i)], fill=(128, 128, 128), width=1)
        grid_image = grid_image.resize((img_size,img_size))

        return np.array(grid_image)


    def __getitem__(self, index):
        sample = {'data':torch.Tensor(self.data[index]).to(torch.float32),
                  'label':torch.Tensor([self.labels[index]]),
                  'indices': torch.Tensor(self.indices[index]).to(torch.float32)}
        return sample


    def get_distances(self, x1, x2=None):
        if x2 is None:
            x2 = x1
        
        x1_nodes = x1[:,:2].long()
        x2_nodes = x2[:,:2].long()
        
        dists = torch.zeros((x1.shape[0], x2.shape[0]))
        for idx in range(x1.shape[0]):
            for idy in range(x2.shape[0]):
                x1_node = self.node_list[(x1_nodes[idx][0].item(), x1_nodes[idx][1].item())]
                x2_node = self.node_list[(x2_nodes[idy][0].item(), x2_nodes[idy][1].item())]
                dists[idx, idy] = self.all_dists[x1_node, x2_node]
        return dists


class EigenTrainer(SupervisedTrainer):
    
    def setup_data(self, config):
        seed = config.get('seed', 123)
        gridsize = config.get('gridsize', 10) # number of samples
        batch_size = config.get('batch_size', 256)
        shuffle = config.get('shuffle', True)

        self.domain = config.get('domain', 'grid')
        if self.domain == 'grid':
            self.dataset = InvarianceDataset(seed, gridsize)
        elif self.domain == 'graph':
            self.dataset = GraphInvarianceDataset(seed, gridsize)
        elif self.domain == 'image':
            self.dataset = ImageInvarianceDataset(seed, gridsize)
        else:
            raise ValueError(f"Unknown data type {self.domain}")
        
        self.train_data = DataLoader(self.dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle
                                     )
        self.kernel_type = config.get('kernel_type', 'rbf')
        self.kernel = self.dataset.get_kernel(self.train_data.dataset.indices)
        self.get_kernel = self.dataset.get_kernel
        self.get_distances = self.dataset.get_distances
        
    def log_epoch(self, config):
        if self.epoch % self.config['log_freq'] == 0:
            # Training and evaluation loss
            train_loss = self.train_log[-1]['log']['loss'] 
            eval_loss = self.eval_log[-1]['log']['loss']
            # Training and evaluation prediction losses
            train_y = torch.from_numpy(self.train_log[-1]['log']['y'])
            train_preds = torch.from_numpy(self.train_log[-1]['log']['preds'])
            train_pred_loss = torch.nn.functional.mse_loss(train_y, train_preds)
            eval_y = torch.from_numpy(self.eval_log[-1]['log']['y'])
            eval_preds = torch.from_numpy(self.eval_log[-1]['log']['preds'])
            eval_pred_loss = torch.nn.functional.mse_loss(eval_y, eval_preds)

            # Optimizer state
            current_lr = self.model.optimizer.param_groups[0]['lr']
            # Gradient with respect to distractor (eta)
            grad = self.eval_log[-1]['log']['grads']
            if self.domain == 'image':
                # Distractors are the pixels in the padding
                pad_size = self.dataset.pad_size
                eta_grad = abs(grad[:,:,:pad_size,:]).sum(axis=(1,2,3))
                eta_grad += abs(grad[:,:,:,:pad_size]).sum(axis=(1,2,3))
                eta_grad += abs(grad[:,:,-pad_size:,:]).sum(axis=(1,2,3))
                eta_grad += abs(grad[:,:,:,-pad_size:]).sum(axis=(1,2,3))
                eta_grad /= abs(grad).sum(axis=(1,2,3)) # normalize wrt total gradient
                eta_grad = eta_grad.mean() # mean over batches 
            else:
                # Distractor is the last element of the input vector
                grad = grad.reshape(grad.shape[0], -1)
                eta_grad = (grad[:, -1]/grad.sum(axis=1)).mean()

            # Log scalars
            self.logger.log(log_dict={
                'trainer_step': self.epoch,
                # Training Metrics
                'train/loss': train_loss,
                'train/pred_loss': train_pred_loss,
                'eval/pred_loss': eval_pred_loss,
                # Evaluation Metrics
                'eval/loss': eval_loss, 
                'eval/protected_grad': eta_grad, 
                # Optimizer
                'optim/lr': current_lr                   
                })


            # Log linecharts: train and eval predictions versus targets
            train_data = {
                'x': [range(train_y.shape[0])]*2,
                'y': [train_preds.reshape(-1,1), train_y.reshape(-1,1)],
                'keys': ["Estimated", "True"],
                'title': f"Predictions [Epoch {self.epoch}]"
            }
            self.logger.log_linechart(key=f'train/preds', data=train_data)
            eval_data = {
                'x': [range(eval_y.shape[0])]*2,
                'y': [eval_preds.reshape(-1,1), eval_y.reshape(-1,1)],
                'keys': ["Estimated", "True"],
                'title': f"Predictions [Epoch {self.epoch}]"
            }
            self.logger.log_linechart(key=f'eval/preds', data=eval_data)

            # Log images: observations and gradients
            x = torch.from_numpy(self.eval_log[-1]['log']['x'])
            encodings = torch.from_numpy(self.eval_log[-1]['log']['encodings'])
            indices = torch.from_numpy(self.eval_log[-1]['log']['indices'])
            grads = torch.from_numpy(self.eval_log[-1]['log']['grads'])
            # Normalize grads for visualization
            grads = (grads - grads.min())/(grads.max() - grads.min())*255.0
            kernel  = self.get_kernel(indices.to('cpu'))
            _, true_eigvecs = torch.lobpcg(kernel, k=encodings.shape[-1], largest=True)
            # eigvals, true_eigvecs = torch.linalg.eigh(kernel.to('cpu'))
            # One plot for every eigenvector
            for ide in range(5):
                encodings_ide = encodings[:, ide]
                true_eigvec = true_eigvecs[:, ide]
                if self.domain == 'image':
                    # Plot some of the inputs
                    plot_idx = np.random.choice(range(x.shape[0]), 4)
                    img = make_grid([x[idx, :] for idx in plot_idx], 
                                    nrow=2
                                    )
                    self.logger.log_image(f'eval/obs', img, image_mode='chw')
                    # Plot their grads
                    img_grad = make_grid([grads[idx, :] for idx in plot_idx], 
                                    nrow=2
                                    )
                    self.logger.log_image(f'eval/grad', img_grad, image_mode='chw')


                    # Plot eigenvectors
                    height = width = int(np.sqrt(x.shape[0]))
                    encodings_ide = rearrange(encodings_ide, '(h w)-> h w', w=width).cpu().numpy()
                    encodings_ide = self.make_heatmap(encodings_ide)
                    true_eigvec = rearrange(true_eigvec, '(h w)-> h w', w=width).cpu().numpy()
                    true_eigvec = self.make_heatmap(true_eigvec)
                    img = make_grid([torch.from_numpy(encodings_ide), 
                                    torch.from_numpy(true_eigvec)], 
                                    nrow=1
                                    )
                    self.logger.log_image(f'eval/encodigs_{ide}', img, image_mode='chw')

                else:
                    x_dim = x.shape[1]
                    if x_dim == 2:
                        grid_data = x[:,:1]
                        data = {
                            'x': [grid_data]*2,
                            'y': [encodings_ide, true_eigvec],
                            'keys': ["Estimated", "True"],
                            'title': f"Eigenfunction {ide} [Epoch {self.epoch}]"
                        }
                        self.logger.log_linechart(key=f'eval/eigvec_{ide}', data=data)
                    elif x_dim == 3:
                        grid_data = x[:,:2]
                        height = width = int(np.sqrt(x.shape[0]))
                        encodings_ide = rearrange(encodings_ide, '(h w)-> h w', w=width).cpu().numpy()
                        encodings_ide = self.make_heatmap(encodings_ide)
                        true_eigvec = rearrange(true_eigvec, '(h w)-> h w', w=width).cpu().numpy()
                        true_eigvec = self.make_heatmap(true_eigvec)
                        img = make_grid([torch.from_numpy(encodings_ide), 
                                        torch.from_numpy(true_eigvec)], 
                                        nrow=1
                                        )
                        self.logger.log_image(f'eval/encodigs_{ide}', img, image_mode='chw')
       


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
               

class EigenEvaluator(SupervisedEvaluator):

    def setup_data(self, config):
        seed = config.get('seed', 321)
        gridsize = config.get('gridsize', 8) # number of samples
        batch_size = config.get('batch_size', 256)
        shuffle = config.get('shuffle', False)

        self.domain = config.get('domain', 'grid')
        if self.domain == 'grid':
            self.dataset = InvarianceDataset(seed, gridsize)
        elif self.domain == 'graph':
            self.dataset = GraphInvarianceDataset(seed, gridsize)
        elif self.domain == 'image':
            self.dataset = ImageInvarianceDataset(seed, gridsize)
        else:
            raise ValueError(f"Unknown data type {self.domain}")
        
        self.dataloader = DataLoader(self.dataset,
                                     batch_size=batch_size,
                                     shuffle=shuffle
                                     )
        self.kernel = self.dataset.get_kernel(self.dataset.indices)
        self.get_kernel = self.dataset.get_kernel
        self.get_distances = self.dataset.get_distances




