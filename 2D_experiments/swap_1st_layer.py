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
from siren_ff import SIRENFF, input_mapping, MLP, get_embedder
from neural_video import train_model

import matplotlib.pyplot as plt
REVERSE = -1
FORWARD = 1

do_color_scheme_transfer = False
do_refine = False
# base_frame = 10
base_frame = 500
direction = FORWARD
refine_iter = train_iters

# read video
if is_vid:
    reader = FFmpegReader(os.path.join(vid_path, vid_name))
    n_frames, h, w, c = reader.getShape()
    data_iterator = reader.nextFrame()
else:
    frame_fn_list = sorted(glob.glob(os.path.join(vid_path, 'frames', '*.png')))
    tmp_img = np.array(Image.open(frame_fn_list[0]))
    h, w = tmp_img.shape[:2]
    def frame_reader():
        for frame_fn in frame_fn_list:
            img = np.array(Image.open(frame_fn))
            yield img
    data_iterator = frame_reader()
image_resolution = (h//img_downsample, w//img_downsample)

# create models. make sure you have the trained weights already
model_dir = os.path.join(vid_path, 'models_incremental')
if use_nerf_pe:
    B, _ = get_embedder(mapping_size)
else:
    B = torch.load(os.path.join(model_dir, '0001_B.pth'))
if no_siren_only_mlp:
    model_base = MLP(*network_size).cuda()
    model_next = MLP(*network_size).cuda()
    model_mix = MLP(*network_size).cuda()
else:
    model_base = SIRENFF(*network_size).cuda()
    model_next = SIRENFF(*network_size).cuda()
    model_mix = SIRENFF(*network_size).cuda()
model_fn_list = sorted(glob.glob(os.path.join(model_dir, '*.pth')))


# def compare_weight_diff(model1, model2):
#     weights1 = model1.state_dict()
#     weights2 = model2.state_dict()
#
#     diff_layer_mean_dict = {}
#     diff_total = 0.
#     n_params_total = 0
#     for layer in weights1.keys():
#         w_diff = (weights1[layer] - weights2[layer]).abs()
#         diff_layer_mean_dict[layer] = w_diff.mean()
#         diff_total += w_diff.sum()
#         n_params_total += w_diff.view(-1).shape[0]
#     diff_total_mean = diff_total / n_params_total
#
#     return diff_layer_mean_dict, diff_total_mean


if do_color_scheme_transfer:
    raw_result_dir = os.path.join(vid_path, 'imgs_color_transfer_before_training'+('_reverse' if direction == REVERSE else ''))
    refined_result_dir = os.path.join(vid_path, 'imgs_color_transfer'+'_reverse'+('_reverse' if direction == REVERSE else ''))
else:
    raw_result_dir = os.path.join(vid_path, 'imgs_swap_1st_raw'+('_reverse' if direction == REVERSE else ''))
    refined_result_dir = os.path.join(vid_path, 'imgs_swap_1st_refined'+('_reverse' if direction == REVERSE else ''))

if os.path.exists(raw_result_dir) or os.path.exists(refined_result_dir):  #or os.path.exists(quantized_result_dir):
    assert False, "make sure you clean previous data"
else:
    os.makedirs(raw_result_dir)
    os.makedirs(refined_result_dir)

model_fn_base = os.path.join(model_dir, f'{base_frame:04d}.pth')
loss_fn = torch.nn.MSELoss()
for frame_i, frame in enumerate(data_iterator):

    if direction != REVERSE and frame_i < base_frame:
        continue
    elif direction == REVERSE and frame_i not in [base_frame-4,base_frame-3,base_frame-2,base_frame-1]:
        continue

    # making data
    ds = ImageDataset(frame, (image_resolution[1], image_resolution[0]))
    grid, image = ds[0]
    grid = grid.unsqueeze(0).cuda()
    image = image.unsqueeze(0).cuda()
    test_data = (grid, image)
    train_data = (grid, image)

    model_fn_next = os.path.join(model_dir, f'{frame_i:04d}.pth')
    model_base.load_state_dict(torch.load(model_fn_base))
    model_next.load_state_dict(torch.load(model_fn_next))
    # diff_layer_mean_dict, diff_total_mean = compare_weight_diff(model_cur, model_next)

    # swap new 1st layers in to base_frame model
    if do_color_scheme_transfer:
        if frame_i == (base_frame + 1):
            model_mix.load_state_dict(torch.load(model_fn_base))
    else:
        model_mix.load_state_dict(torch.load(model_fn_base))
        if no_siren_only_mlp:
            model_mix.state_dict()['first_layer.weight'].copy_(model_next.state_dict()['first_layer.weight'])
            model_mix.state_dict()['first_layer.bias'].copy_(model_next.state_dict()['first_layer.bias'])
        else:
            model_mix.state_dict()['first_layer.linear.weight'].copy_(model_next.state_dict()['first_layer.linear.weight'])
            model_mix.state_dict()['first_layer.linear.bias'].copy_(model_next.state_dict()['first_layer.linear.bias'])

    # visualize structure swap
    with torch.no_grad():
        # # render original base frame
        # out_cur, _ = model_base(input_mapping(test_data[0], B, use_nerf_pe=use_nerf_pe))
        # # render original new frame
        # out_next, _ = model_next(input_mapping(test_data[0], B, use_nerf_pe=use_nerf_pe))
        # render structure swap (base frame with new 1st layer)
        out_mix, _ = model_mix(input_mapping(test_data[0], B, use_nerf_pe=use_nerf_pe))
    mix_loss = loss_fn(out_mix, image)
    mix_psnr = - 10 * torch.log10(2 * mix_loss).item()
    print(f"raw c-{base_frame} s-{frame_i} : {mix_psnr:.6f}db")
    torchvision.utils.save_image(out_mix.permute(0,3,1,2), os.path.join(raw_result_dir, f"{base_frame:04d}_{frame_i:04d}_{mix_psnr:.6f}.jpeg"))

    # refine the swapped-in 1st layer
    if do_refine:
        optim = torch.optim.Adam(model_mix.parameters(), lr=learning_rate)
        for i, param in enumerate(model_mix.parameters()):
            if i<2:
                continue
            param.requires_grad = False

        out_mix_refined = train_model(model_mix, optim, loss_fn, refine_iter, B, use_nerf_pe=use_nerf_pe,
                                      train_data=train_data, test_data=(grid, image), device='cuda')
        model_mix_1st_layer_refined = out_mix_refined['model']
        img_mix_refined = out_mix_refined['img']
        mix_loss_refined = loss_fn(img_mix_refined, image)
        mix_psnr_refined = - 10 * torch.log10(2 * mix_loss_refined).item()
        print(f"optimized: {mix_psnr_refined:.6f}db\n")
        torchvision.utils.save_image(img_mix_refined.permute(0, 3, 1, 2),
                                     os.path.join(refined_result_dir, f"{base_frame:04d}_{frame_i:04d}_{mix_psnr_refined:.6f}.jpeg"))

    # check only first layer changed
    if frame_i == (base_frame+1):
        if no_siren_only_mlp:
            print((model_mix.first_layer.weight - model_base.first_layer.weight).abs().max())
            for l in range(3):
                print((model_mix.mid_layers[l].weight - model_base.mid_layers[l].weight).abs().max())
            print((model_mix.final_layer.weight - model_base.final_layer.weight).abs().max())
        else:
            print((model_mix.first_layer.linear.weight
                   - model_base.first_layer.linear.weight).abs().max())
            for l in range(3):
                print((model_mix.mid_layers[l].linear.weight
                       - model_base.mid_layers[l].linear.weight).abs().max())
            print((model_mix.final_layer.linear.weight
                   - model_base.final_layer.linear.weight).abs().max())





