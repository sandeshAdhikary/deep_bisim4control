# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans, KMeans
import math

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        # out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        # self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)

        # Infer output dim 
        self.outputs = dict()
        with torch.no_grad():
            dummy = torch.zeros((1, *obs_shape))
            dummy = self.forward_conv(dummy)
            out_dim = dummy.shape[-1]
            self.outputs = dict() # Reset self.outputs

        self.fc = nn.Linear(out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs_input):
        obs = obs_input / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs_input, detach=False):
        h = self.forward_conv(obs_input)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)

        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

        #TODO: The paper says to use a tanh activation here, but the code doesn't
        h_fc = torch.tanh(h_fc)
        
        return out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        # for k, v in self.outputs.items():
        #     L.log_histogram('train_encoder/%s_hist' % k, v, step)
        #     if len(v.shape) > 2:
        #         L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)


class PixelEncoderL2Norm(PixelEncoder):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs):
        super().__init__(obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, **kwargs)

        # Register L2_norm params
        self.l2_norm_layer = True
        self.l2_normalize_over = [0]
        self.l2_norm_momentum = 0.9
        self.register_buffer('eigennorm', torch.ones(self.feature_dim))
        self.register_buffer('num_calls', torch.Tensor([0]))

    def forward(self, obs_input, detach=False):
        out = super().forward(obs_input, detach)
        
        # Apply final L2 norm layer
        if self.l2_norm_layer:
            if self.training:
                norm_ = out.norm(dim=self.l2_normalize_over) / math.sqrt(np.prod([out.shape[dim] for dim in self.l2_normalize_over]))
                with torch.no_grad():
                    if self.num_calls == 0:
                        self.eigennorm.copy_(norm_.data)
                    else:
                        self.eigennorm.mul_(self.l2_norm_momentum).add_(
                            norm_.data, alpha = 1-self.l2_norm_momentum)
                    self.num_calls += 1
            else:
                norm_ = self.eigennorm
            out = out / norm_

        return out



class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1, **kwargs):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=stride))

        out_dims = 100  # if defaults change, adjust this as needed
        self.fc = nn.Linear(num_filters * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class PixelEncoderCarla098(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1, **kwargs):
        super(PixelEncoder, self).__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList()
        self.convs.append(nn.Conv2d(obs_shape[0], 64, 5, stride=2))
        self.convs.append(nn.Conv2d(64, 128, 3, stride=2))
        self.convs.append(nn.Conv2d(128, 256, 3, stride=2))
        self.convs.append(nn.Conv2d(256, 256, 3, stride=2))

        out_dims = 56  # 3 cameras
        # out_dims = 100  # 5 cameras
        self.fc = nn.Linear(256 * out_dims, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


class VectorEncoder(nn.Module):
    """
    Simple NN (non-convolutional) encoder for observations
    """
    def __init__(self, obs_shape, feature_dim, num_layers=None, num_filters=None, stride=None, **kwargs):
        super().__init__()

        assert len(obs_shape) == 1

        self.input_shape = obs_shape[0]
        self.feature_dim = feature_dim
        self.num_layers = 2

        self.fc = nn.Sequential(
            nn.Linear(self.input_shape, 4*self.input_shape),
            nn.ReLU(),
            nn.Linear(4*self.input_shape, 128),
            nn.ReLU(),
        )
        self.output_layer = nn.Linear(128, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


    def forward(self, obs_input, detach=False):
        out = self.fc(obs_input)
        out = self.output_layer(out)
        out = self.ln(out)
        #TODO: The paper says to use a tanh activation here, but the code doesn't
        out = torch.tanh(out)
        return out
    
    def copy_conv_weights_from(self, source):
        #TODO: This was originally meant for pixel encoders; Currently equating conv_layers -> fc_layers for vec encoder
        # tie everything except the last output layer
        for i, source_layer in enumerate(source.fc):
            if isinstance(source_layer, nn.modules.linear.Linear):
                tie_weights(src=source_layer, trg=self.fc[i])

    def log(self, L, step, log_freq):
        pass


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, stride, **kwargs):
        super().__init__()

        assert len(obs_shape) == 1
        self.feature_dim = obs_shape[0]

    def forward(self, obs, detach=False):
        return obs

    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


##### Encoder with clusters
# class ClusterEncoder(nn.Module):
#     def __init__(self, encoder, num_clusters, batch_size, seed=123):
#         """
#         Wrapper around an encoder that adds a final clustering layer
#         The output of the original encoder is provided to the clusterer
#         to yield a final feature (with num_clusters dimensions) of soft
#         cluster assignments
#         """
#         super().__init__()
#         self._encoder = encoder
#         self.parent_attr = "Parent's attribute"
#         self.num_clusters = self.output_dim = num_clusters
#         self.batch_size = batch_size
#         self.seed = seed
#         self.clusterer = MiniBatchKMeans(n_clusters = num_clusters,
#                                          init='k-means++',
#                                          n_init='auto',
#                                          batch_size=batch_size,
#                                          random_state=seed
#                                          )
#         self.centroids = self._init_centroids()

#     def _init_centroids(self):
#         return torch.rand((self.num_clusters, self._encoder.feature_dim))
                                

#     def forward(self, obs_input, detach=False):
#         h = self._encoder(obs_input, detach)
#         features = torch.cdist(h, self.centroids.to(obs_input.device), p=2)
#         features = torch.exp(-features**2)
#         return features

#     def update_centroids(self, features, reset=False):
#         with torch.no_grad():
#             if reset:
#                 # Reset the clusterer; initialize at earlier centroids
#                 self.clusterer = MiniBatchKMeans(n_clusters = self.num_clusters,
#                                     init=self.centroids,
#                                     n_init='auto',
#                                     batch_size=self.batch_size,
#                                     random_state=self.seed
#                                     )
#             # Update centroids
#             self.clusterer.partial_fit(features.detach().cpu().numpy())
#             self.centroids = torch.from_numpy(self.clusterer.cluster_centers_).to(features.device)

#     def __getattr__(self, name):
#         if '_parameters' in self.__dict__:
#             _parameters = self.__dict__['_parameters']
#             if name in _parameters:
#                 return _parameters[name]
#         if '_buffers' in self.__dict__:
#             _buffers = self.__dict__['_buffers']
#             if name in _buffers:
#                 return _buffers[name]
#         if '_modules' in self.__dict__:
#             modules = self.__dict__['_modules']
#             if name in modules:
#                 return modules[name]
        
#         # Get attribute from child-encoder
#         if hasattr(self._encoder, name):
#             return getattr(self._encoder, name)

#         raise AttributeError("'{}' object has no attribute '{}'".format(
#             type(self).__name__, name))

# class ClusterPixelEncoder(ClusterEncoder):
#     def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, num_clusters=3, batch_size=128, seed=123):
#         encoder = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, stride)
#         super().__init__(encoder, num_clusters, batch_size, seed=seed)

# class ClusterPixelEncoderCarla096(ClusterEncoder):
#     def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, num_clusters=3, batch_size=128, seed=123):
#         encoder = PixelEncoderCarla096(obs_shape, feature_dim, num_layers, num_filters, stride)
#         super().__init__(encoder, num_clusters, batch_size, seed=seed)

# class ClusterPixelEncoderCarla098(ClusterEncoder):
#     def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, num_clusters=3, batch_size=128, seed=123):
#         encoder = PixelEncoderCarla098(obs_shape, feature_dim, num_layers, num_filters, stride)
#         super().__init__(encoder, num_clusters, batch_size, seed=seed)

# class ClusterIdentityEncoder(ClusterEncoder):
#     def __init__(self, obs_shape, feature_dim, num_layers, num_filters, stride, num_clusters=3, batch_size=128, seed=123):
#         encoder = IdentityEncoder(obs_shape, feature_dim, num_layers, num_filters, stride)
#         super().__init__(encoder, num_clusters, batch_size, seed=seed)

# class ClusterVectorEncoder(ClusterEncoder):
#     def __init__(self, obs_shape, feature_dim, num_layers, num_filters, stride, num_clusters=3, batch_size=128, seed=123):
#         encoder = VectorEncoder(obs_shape, feature_dim, num_layers, num_filters, stride)
#         super().__init__(encoder, num_clusters, batch_size, seed=seed)


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixel_l2': PixelEncoderL2Norm, 
                    #    'pixel_cluster': ClusterPixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                    #    'pixelCarla096_cluster': ClusterPixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                    #    'pixelCarla098_cluster': ClusterPixelEncoderCarla098,
                       'identity': IdentityEncoder,
                    #    'identity_cluster': ClusterIdentityEncoder,
                       'vector': VectorEncoder,
                    #    'vector_cluster': ClusterVectorEncoder
                       }

_CLUSTER_ENCODERS = {}

# _CLUSTER_ENCODERS = {
#                     'pixel_cluster': ClusterPixelEncoder,
#                     'pixelCarla096_cluster': ClusterPixelEncoderCarla096,
#                     'pixelCarla098_cluster': ClusterPixelEncoderCarla098,
#                     'identity_cluster': ClusterIdentityEncoder,
#                     'vector_cluster': ClusterVectorEncoder
# }


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride, output_dim=None
):
    
    # If output_dim not specified, assume it is feature_dim
    output_dim = output_dim or feature_dim

    assert encoder_type in _AVAILABLE_ENCODERS

    if encoder_type in _CLUSTER_ENCODERS:
        # Set cluster dim to be output dim
        return _CLUSTER_ENCODERS[encoder_type](
                obs_shape, feature_dim, num_layers, num_filters, stride, num_clusters=output_dim
            )
    else:
        return _AVAILABLE_ENCODERS[encoder_type](
            obs_shape, feature_dim, num_layers, num_filters, stride
        )
