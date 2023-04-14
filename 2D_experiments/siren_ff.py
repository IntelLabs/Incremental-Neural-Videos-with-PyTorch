import torch, torchvision, glob, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
from PIL import Image
from skvideo.io.ffmpeg import FFmpegReader

from dataset import ImageDataset

import matplotlib.pyplot as plt


def get_embedder(multires):
    embed_kwargs = {
        'include_input': True,
        'input_dims': 3,
        'max_freq_log2': multires - 1,
        'num_freqs': multires,
        'log_sampling': True,
        'periodic_fns': [torch.sin, torch.cos],
    }

    embedder_obj = Embedder(**embed_kwargs)
    embed = lambda x, eo=embedder_obj: eo.embed(x)
    return embed, embedder_obj.out_dim


class Embedder:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.create_embedding_fn()

    def create_embedding_fn(self):
        embed_fns = []
        d = self.kwargs['input_dims']
        out_dim = 0
        if self.kwargs['include_input']:
            embed_fns.append(lambda x: x)
            out_dim += d

        max_freq = self.kwargs['max_freq_log2']
        N_freqs = self.kwargs['num_freqs']

        if self.kwargs['log_sampling']:
            freq_bands = 2. ** torch.linspace(0., max_freq, steps=N_freqs)
        else:
            freq_bands = torch.linspace(2. ** 0., 2. ** max_freq, steps=N_freqs)

        for freq in freq_bands:
            for p_fn in self.kwargs['periodic_fns']:
                embed_fns.append(lambda x, p_fn=p_fn, freq=freq: p_fn(x * freq))
                out_dim += d

        self.embed_fns = embed_fns
        self.out_dim = out_dim

    def embed(self, inputs):
        return torch.cat([fn(inputs) for fn in self.embed_fns], -1)


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False, is_mid=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.is_mid = is_mid
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def input_mapping(x, B, use_nerf_pe):
    '''

    Parameters
    ----------
    x - Nxd input coordinates, d is dimension (e.g. 3 for nerf)
    B - if use_nerf_pe, this is a list of sin and cos encoding funcs of different freqs
                        output is thus N x 2*d
        else, this is a fourier feature k x d matrix with k random components,
                        output is N x 2*k
    use_nerf_pe

    Returns
    -------

    '''
    if B is None:
        return x
    elif not use_nerf_pe:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)
    else:
        return B(x)


class SIRENFF(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim=3):
        """
        """
        super(SIRENFF, self).__init__()
        self.first_layer = SirenLayer(input_dim, hidden_dim, is_first=True)
        self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim, hidden_dim)
                                         for i in range(1, num_layers - 1)])
        self.final_layer = SirenLayer(hidden_dim, output_dim, is_last=True)

    def forward(self, x):
        x = self.first_layer(x)
        y = None
        for i, layer in enumerate(self.mid_layers):
            x = layer(x)
        x = self.final_layer(x)
        return x, y


class MLP(nn.Module):
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim=3):
        """
        """
        super(MLP, self).__init__()
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim, hidden_dim) for i in range(1, num_layers - 1)])
        self.final_layer = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.first_layer(x))
        y = None
        for i, layer in enumerate(self.mid_layers):
            x = F.relu(layer(x))
        x = self.final_layer(x)
        return x, y