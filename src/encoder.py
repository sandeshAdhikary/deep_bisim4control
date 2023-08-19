# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import MiniBatchKMeans, KMeans

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None):
        super().__init__()

        assert len(obs_shape) == 3

        self.feature_dim = feature_dim
        self.num_layers = num_layers

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = {2: 39, 4: 35, 6: 31}[num_layers]
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()

    def reparameterize(self, mu, logstd):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        out = self.ln(h_fc)
        self.outputs['ln'] = out

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


class PixelEncoderCarla096(PixelEncoder):
    """Convolutional encoder of pixels observations."""
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
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
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=1):
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
    def __init__(self, obs_shape, feature_dim, num_layers=None, num_filters=None, stride=None):
        super().__init__()

        assert len(obs_shape) == 1

        self.input_shape = obs_shape[0]
        self.feature_dim = feature_dim
        # self.num_layers = num_layers

        self.fc = nn.Sequential(
            nn.Linear(self.input_shape, 4*self.input_shape),
            nn.ReLU(),
            nn.Linear(4*self.input_shape, 128),
            nn.ReLU(),
            nn.Linear(128, self.feature_dim),
        )
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()


    def forward(self, obs, detach=False):
        return self.ln(self.fc(obs))
    
    def copy_conv_weights_from(self, source):
        pass

    def log(self, L, step, log_freq):
        pass


class IdentityEncoder(nn.Module):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, stride):
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
class ClusterEncoder(nn.Module):
    def __init__(self, encoder, num_clusters, batch_size, seed=123):
        """
        Wrapper around an encoder that adds a final clustering layer
        The output of the original encoder is provided to the clusterer
        to yield a final feature (with num_clusters dimensions) of soft
        cluster assignments
        """
        super().__init__()
        self._encoder = encoder
        self.parent_attr = "Parent's attribute"
        self.num_clusters = num_clusters
        self.batch_size = batch_size
        self.seed = seed
        self.clusterer = MiniBatchKMeans(n_clusters = num_clusters,
                                         init='k-means++',
                                         n_init='auto',
                                         batch_size=batch_size,
                                         random_state=seed
                                         )
        self.centroids = self._init_centroids()

    def _init_centroids(self):
        return torch.rand((self.num_clusters, self._encoder.feature_dim))
                                


    def forward(self, obs, detach=False):
        h = self._encoder(obs, detach)
        features = torch.cdist(h, self.centroids.to(obs.device), p=2)
        features = torch.exp(-features**2)
        return features

    def update_centroids(self, features, reset=False):
        with torch.no_grad():
            if reset:
                # Reset the clusterer; initialize at earlier centroids
                self.clusterer = MiniBatchKMeans(n_clusters = self.num_clusters,
                                    init=self.centroids,
                                    n_init='auto',
                                    batch_size=self.batch_size,
                                    random_state=self.seed
                                    )
            # Update centroids
            self.clusterer.partial_fit(features.detach().cpu().numpy())
            self.centroids = torch.from_numpy(self.clusterer.cluster_centers_).to(features.device)

    def __getattr__(self, name):
        if '_parameters' in self.__dict__:
            _parameters = self.__dict__['_parameters']
            if name in _parameters:
                return _parameters[name]
        if '_buffers' in self.__dict__:
            _buffers = self.__dict__['_buffers']
            if name in _buffers:
                return _buffers[name]
        if '_modules' in self.__dict__:
            modules = self.__dict__['_modules']
            if name in modules:
                return modules[name]
        
        # Get attribute from child-encoder
        if hasattr(self._encoder, name):
            return getattr(self._encoder, name)

        raise AttributeError("'{}' object has no attribute '{}'".format(
            type(self).__name__, name))

class ClusterPixelEncoder(ClusterEncoder):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, num_clusters=3, batch_size=128, seed=123):
        encoder = PixelEncoder(obs_shape, feature_dim, num_layers, num_filters, stride)
        super().__init__(encoder, num_clusters, batch_size, seed=seed)

class ClusterPixelEncoderCarla096(ClusterEncoder):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, num_clusters=3, batch_size=128, seed=123):
        encoder = PixelEncoderCarla096(obs_shape, feature_dim, num_layers, num_filters, stride)
        super().__init__(encoder, num_clusters, batch_size, seed=seed)

class ClusterPixelEncoderCarla098(ClusterEncoder):
    def __init__(self, obs_shape, feature_dim, num_layers=2, num_filters=32, stride=None, num_clusters=3, batch_size=128, seed=123):
        encoder = PixelEncoderCarla098(obs_shape, feature_dim, num_layers, num_filters, stride)
        super().__init__(encoder, num_clusters, batch_size, seed=seed)

class ClusterIdentityEncoder(ClusterEncoder):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, stride, num_clusters=3, batch_size=128, seed=123):
        encoder = IdentityEncoder(obs_shape, feature_dim, num_layers, num_filters, stride)
        super().__init__(encoder, num_clusters, batch_size, seed=seed)

class ClusterVectorEncoder(ClusterEncoder):
    def __init__(self, obs_shape, feature_dim, num_layers, num_filters, stride, num_clusters=3, batch_size=128, seed=123):
        encoder = VectorEncoder(obs_shape, feature_dim, num_layers, num_filters, stride)
        super().__init__(encoder, num_clusters, batch_size, seed=seed)


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder,
                       'pixel_cluster': ClusterPixelEncoder,
                       'pixelCarla096': PixelEncoderCarla096,
                       'pixelCarla096_cluster': ClusterPixelEncoderCarla096,
                       'pixelCarla098': PixelEncoderCarla098,
                       'pixelCarla098_cluster': ClusterPixelEncoderCarla098,
                       'identity': IdentityEncoder,
                       'identity_cluster': ClusterIdentityEncoder,
                       'vector': VectorEncoder,
                       'vector_cluster': ClusterVectorEncoder
                       }


def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, stride
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, stride
    )
