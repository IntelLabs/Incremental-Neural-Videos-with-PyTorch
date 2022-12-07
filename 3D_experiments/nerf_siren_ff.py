import torch, torchvision, glob, os
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
# from tqdm import tqdm
# from PIL import Image
# from skvideo.io.ffmpeg import FFmpegReader
#
# from dataset import ImageDataset

import matplotlib.pyplot as plt


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
        ######## TODO: first layer 30, others 1 ###########
        if self.is_first:
            self.w0 = 30
        else:
            self.w0 = 1
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
        # if self.is_last:
        #     return x
        # elif self.is_mid:
        #     return nn.functional.relu(x)
        # else:
        #     return torch.sin(self.w0 * x)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class NeRF_SIRENFF_old(nn.Module):
    def __init__(self, num_layers, input_dim, input_view_dim, hidden_dim, skip=-1):
        """
        """
        super(NeRF_SIRENFF_old, self).__init__()
        self.skip = skip
        self.input_dim = input_dim
        self.input_view_dim = input_view_dim
        self.first_layer = SirenLayer(input_dim, hidden_dim, is_first=True)
        self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim+input_dim, hidden_dim) if i == skip
                                         else SirenLayer(hidden_dim, hidden_dim)
                                         for i in range(num_layers - 1)])
        # self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim, hidden_dim),
        #                                  SirenLayer(hidden_dim, hidden_dim),
        #                                  SirenLayer(hidden_dim, hidden_dim)])
        # self.mid_supervision = SirenLayer(hidden_dim, 3, is_last=True)
        self.alpha_layer = SirenLayer(hidden_dim, 1, is_last=True)
        self.rgb_layer_1 = SirenLayer(hidden_dim, hidden_dim)
        self.rgb_layer_2 = SirenLayer(hidden_dim+input_view_dim, hidden_dim//2)
        self.rgb_layer_3 = SirenLayer(hidden_dim//2, 3, is_last=True)

    def forward(self, input):
        input_pts, input_viewdir = torch.split(input, [self.input_dim, self.input_view_dim], dim=-1)
        x = self.first_layer(input_pts)
        y = None
        for i, layer in enumerate(self.mid_layers):
            if i == self.skip:
                x = torch.cat([input_pts, x], dim=-1)
            x = layer(x)
        a = self.alpha_layer(x)

        x = self.rgb_layer_1(x)
        x = self.rgb_layer_2(torch.cat([input_viewdir, x], dim=-1))
        rgb = self.rgb_layer_3(x)

        return torch.cat([rgb,a], dim=-1)


class NeRF_FF(nn.Module):
    def __init__(self, num_layers, input_dim, input_view_dim, hidden_dim, skip=-1):
        """
        """
        super(NeRF_FF, self).__init__()
        self.skip = skip
        self.input_dim = input_dim
        self.input_view_dim = input_view_dim
        self.first_layer = nn.Linear(input_dim, hidden_dim)
        self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim+input_dim, hidden_dim) if i == skip
                                         else nn.Linear(hidden_dim, hidden_dim)
                                         for i in range(num_layers - 1)])
        # self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim, hidden_dim),
        #                                  SirenLayer(hidden_dim, hidden_dim),
        #                                  SirenLayer(hidden_dim, hidden_dim)])
        # self.mid_supervision = SirenLayer(hidden_dim, 3, is_last=True)
        self.alpha_layer = nn.Linear(hidden_dim, 1)
        self.rgb_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        self.rgb_layer_2 = nn.Linear(hidden_dim+input_view_dim, hidden_dim//2)
        self.rgb_layer_3 = nn.Linear(hidden_dim//2, 3)

    def forward(self, input):
        input_pts, input_viewdir = torch.split(input, [self.input_dim, self.input_view_dim], dim=-1)
        x = F.relu(self.first_layer(input_pts))
        y = None
        for i, layer in enumerate(self.mid_layers):
            if i == self.skip:
                x = torch.cat([input_pts, x], dim=-1)
            x = F.relu(layer(x))
        a = self.alpha_layer(x)

        x = self.rgb_layer_1(x)
        x = F.relu(self.rgb_layer_2(torch.cat([input_viewdir, x], dim=-1)))
        rgb = self.rgb_layer_3(x)

        return torch.cat([rgb,a], dim=-1)


# with SIREN weight initial
# class NeRF_SIRENFF(nn.Module):
#     def __init__(self, num_layers, input_dim, input_view_dim, hidden_dim, skip=-1):
#         """
#         """
#         super(NeRF_SIRENFF, self).__init__()
#         self.skip = skip
#         self.input_dim = input_dim
#         self.input_view_dim = input_view_dim
#         self.first_layer = SirenLayer(input_dim, hidden_dim, is_first=True)
#         self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim, hidden_dim)
#                                          for i in range(num_layers - 1)])
#         self.alpha_layer = nn.Linear(hidden_dim, 1)
#         self.rgb_layer = nn.Linear(hidden_dim, 3)
#
#         # self.rgb_layer_1 = nn.Linear(hidden_dim, hidden_dim)
#         # self.rgb_layer_2 = SirenLayer(hidden_dim, hidden_dim // 2)
#         # self.rgb_layer_3 = nn.Linear(hidden_dim//2, 3)
#
#     def forward(self, input):
#         input_pts, input_viewdir = torch.split(input, [self.input_dim, self.input_view_dim], dim=-1)
#         x = self.first_layer(input_pts)
#         y = None
#         for i, layer in enumerate(self.mid_layers):
#             x = layer(x)
#         a = F.relu(self.alpha_layer(x))
#         rgb = F.sigmoid(self.rgb_layer(x))
#
#         # x = self.rgb_layer_1(x)
#         # x = self.rgb_layer_2(x)
#         # rgb = F.sigmoid(self.rgb_layer_3(x))
#
#         return torch.cat([rgb, a], -1)

# class NeRF_SIRENFF(nn.Module):
#     def __init__(self, num_layers, input_dim, input_view_dim, hidden_dim, skip=-1):
#         """
#         """
#         super(NeRF_SIRENFF, self).__init__()
#         self.skip = skip
#         self.input_dim = input_dim
#         self.input_view_dim = input_view_dim
#         self.first_layer = nn.Linear(input_dim, hidden_dim)
#         self.mid_layers = nn.ModuleList([nn.Linear(hidden_dim+input_dim, hidden_dim) if i == skip
#                                          else nn.Linear(hidden_dim, hidden_dim)
#                                          for i in range(num_layers - 1)])
#         # self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim, hidden_dim),
#         #                                  SirenLayer(hidden_dim, hidden_dim),
#         #                                  SirenLayer(hidden_dim, hidden_dim)])
#         # self.mid_supervision = SirenLayer(hidden_dim, 3, is_last=True)
#         self.alpha_layer = nn.Linear(hidden_dim, 1)
#         self.rgb_layer_1 = nn.Linear(hidden_dim, hidden_dim)
#         self.rgb_layer_2 = nn.Linear(hidden_dim+input_view_dim, hidden_dim//2)
#         self.rgb_layer_3 = nn.Linear(hidden_dim//2, 3)
#
#     def forward(self, input):
#         input_pts, input_viewdir = torch.split(input, [self.input_dim, self.input_view_dim], dim=-1)
#         x = torch.sin(self.first_layer(input_pts))
#         y = None
#         for i, layer in enumerate(self.mid_layers):
#             if i == self.skip:
#                 x = torch.cat([input_pts, x], dim=-1)
#             x = torch.sin(layer(x))
#         a = self.alpha_layer(x)
#
#         x = self.rgb_layer_1(x)
#         x = torch.sin(self.rgb_layer_2(torch.cat([input_viewdir, x], dim=-1)))
#         rgb = self.rgb_layer_3(x)
#
#         return torch.cat([rgb,a], dim=-1)

# with SIREN init
class NeRF_SIRENFF(nn.Module):
    def __init__(self, num_layers, input_dim, input_view_dim, hidden_dim, skip=-1):
        """
        """
        super(NeRF_SIRENFF, self).__init__()
        self.skip = skip
        self.input_dim = input_dim
        self.input_view_dim = input_view_dim
        self.first_layer = SirenLayer(input_dim, hidden_dim, is_first=True, w0=30)
        self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim+input_dim, hidden_dim) if i == skip
                                         else SirenLayer(hidden_dim, hidden_dim, w0=30)
                                         for i in range(num_layers - 1)])
        # self.mid_layers = nn.ModuleList([SirenLayer(hidden_dim, hidden_dim),
        #                                  SirenLayer(hidden_dim, hidden_dim),
        #                                  SirenLayer(hidden_dim, hidden_dim)])
        # self.mid_supervision = SirenLayer(hidden_dim, 3, is_last=True)
        self.alpha_layer = SirenLayer(hidden_dim, 1, is_last=True)
        # self.rgb_layer_1 = nn.Linear(hidden_dim, hidden_dim)
        # self.rgb_layer_2 = SirenLayer(hidden_dim+input_view_dim, hidden_dim//2)
        # self.rgb_layer_2 = SirenLayer(hidden_dim, hidden_dim // 2)
        # self.rgb_layer_3 = nn.Linear(hidden_dim//2, 3)
        self.rgb_layer = SirenLayer(hidden_dim, 3, is_last=True)

    def forward(self, input):
        input_pts, input_viewdir = torch.split(input, [self.input_dim, self.input_view_dim], dim=-1)
        x = self.first_layer(input_pts)
        y = None
        for i, layer in enumerate(self.mid_layers):
            if i == self.skip:
                x = torch.cat([input_pts, x], dim=-1)
            x = layer(x)
        a = self.alpha_layer(x)

        # x = self.rgb_layer_1(x)
        # x = self.rgb_layer_2(torch.cat([input_viewdir, x], dim=-1))
        # x = self.rgb_layer_2(x)
        rgb = self.rgb_layer(x)

        return torch.cat([rgb,a], dim=-1)
