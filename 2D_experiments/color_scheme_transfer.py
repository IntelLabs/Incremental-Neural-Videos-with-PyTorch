# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch, shutil, configargparse, glob, torchvision
import numpy as np
import skvideo.io
from skvideo.io.ffmpeg import FFmpegReader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from PIL import Image

from const import *
from demo import gon_model, ImageDataset
from siren_ff import SIRENFF, input_mapping, FFN, get_embedder
from neural_video import train_model

import matplotlib.pyplot as plt

base_folder = "/playpen-ssd/mikewang/incremental_neural_videos/2D_data/color_scheme_transfer"
structure_fn = base_folder+"/mlk_color.jpg"
s_factor = 4
color_fn = base_folder+"/mlk_bw.jpg"
c_factor = 2
expname = 'imgs_color_transfer_5'
train_iters = 1000
refine_iter = 400 #train_iters
extra_iter = 800

use_nerf_pe = False
no_siren_only_mlp = True
if use_nerf_pe:
    mapping_size = 128
    network_size = (5, (mapping_size*2)*2+2, 256)
else:
    mapping_size = 256
    network_size = (5, mapping_size*2, 256)

structure_frame = np.array(Image.open(structure_fn))
if len(structure_frame.shape) == 2:
    structure_frame = np.tile(structure_frame[...,None], (1,1,3))
structure_frame = structure_frame[...,:3]
sh, sw = structure_frame.shape[:2]
struct_res = (sh//s_factor, sw//s_factor)
color_frame = np.array(Image.open(color_fn))
if len(color_frame.shape) == 2:
    color_frame = np.tile(color_frame[...,None], (1,1,3))
color_frame = color_frame[...,:3]
ch, cw = color_frame.shape[:2]
color_res = (ch//c_factor, cw//c_factor)

if use_nerf_pe:
    B, _ = get_embedder(mapping_size)
else:
    # B = torch.load(os.path.join(model_dir, '0000_B.pth'))
    # raise "not implemented yet"
    B = torch.randn((mapping_size, 2)).to("cuda") * 10

if no_siren_only_mlp:
    model = FFN(*network_size).cuda()
else:
    model = SIRENFF(*network_size).cuda()
loss_fn = torch.nn.MSELoss()
vis_dir = os.path.join(base_folder, expname)
os.makedirs(vis_dir, exist_ok=True)


# making data
ds = ImageDataset(color_frame, (color_res[1], color_res[0]))
grid, image = ds[0]
grid = grid.unsqueeze(0).cuda()
image = image.unsqueeze(0).cuda()
test_data = (grid, image)
train_data = (grid, image)

# train color
optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
out_color = train_model(model, optim, loss_fn, train_iters, B, use_nerf_pe=use_nerf_pe,
                        train_data=train_data, test_data=(grid, image), device='cuda')

torchvision.utils.save_image(out_color['img'].permute(0, 3, 1, 2),
                             os.path.join(vis_dir, f"orig_model.png"))

# change structure
for i, param in enumerate(model.parameters()):
    if i<2:
        continue
    param.requires_grad = False

ds = ImageDataset(structure_frame, (struct_res[1], struct_res[0]))
grid, image = ds[0]
grid = grid.unsqueeze(0).cuda()
image = image.unsqueeze(0).cuda()
test_data = (grid, image)
train_data = (grid, image)

out_new_struct = train_model(model, optim, loss_fn, refine_iter, B, use_nerf_pe=use_nerf_pe,
                             train_data=train_data, test_data=(grid, image), device='cuda')

torchvision.utils.save_image(out_new_struct['img'].permute(0, 3, 1, 2),
                             os.path.join(vis_dir, f"color_scheme_transfer_{refine_iter}.png"))

out_new_struct = train_model(model, optim, loss_fn, extra_iter, B, use_nerf_pe=use_nerf_pe,
                             train_data=train_data, test_data=(grid, image), device='cuda')

torchvision.utils.save_image(out_new_struct['img'].permute(0, 3, 1, 2),
                             os.path.join(vis_dir, f"color_scheme_transfer_{extra_iter+refine_iter}.png"))

# check only first layer changed
# if no_siren_only_mlp:
#     print((model_mix.first_layer.weight - model_cur.first_layer.weight).abs().max())
#     for l in range(3):
#         print((model_mix.mid_layers[l].weight - model_cur.mid_layers[l].weight).abs().max())
#     print((model_mix.final_layer.weight - model_cur.final_layer.weight).abs().max())
# else:
#     print((model_mix.first_layer.linear.weight
#            - model_cur.first_layer.linear.weight).abs().max())
#     for l in range(3):
#         print((model_mix.mid_layers[l].linear.weight
#                - model_cur.mid_layers[l].linear.weight).abs().max())
#     print((model_mix.final_layer.linear.weight
#            - model_cur.final_layer.linear.weight).abs().max())

# model_mix_1st_layer_refined = out_mix_refined['model']
# img_mix_refined = out_mix_refined['img']
# mix_loss_refined = loss_fn(img_mix_refined, image)
# mix_psnr_refined = - 10 * torch.log10(2 * mix_loss_refined).item()
# print(f"optimized: {mix_psnr_refined:.6f}db\n")

