import os, sys
import numpy as np
import imageio
import json
import random
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange
import fpzip
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from config_parser import config_parser
from model_helpers import *
from helpers import create_nerf, render, render_path, create_ray_rgb_K_batches
from load_llff import load_llff_data, load_LF_data, load_META_data
from nerf_motion_layer import render_from_view

torch.set_default_tensor_type('torch.cuda.FloatTensor')  # NOTE: very important
parser = config_parser()
args = parser.parse_args()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
total_n_layers = 12


def render_from_view(img_i, render_kwargs_test, images, poses, K, H, W, args):
    # img_i = i_train[0]
    gt_img = torch.Tensor(images[img_i]).to(device)
    pose = poses[img_i, :3, :4]
    with torch.no_grad():
        rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                        **render_kwargs_test)

    psnr = mse2psnr(img2mse(rgb, gt_img))
    return psnr, rgb


batch_size = 30


basedir = '/playpen-ssd/mikewang/incremental_neural_videos/META_data/flame_salmon_1/down_2x/'
expname = 'META_flame_salmon_1_warmup10k_iter10k_s3_stability0_relu_l2_freeze120'
args.mid_freeze_start = 3
frame_a = 120

exp_folder = basedir + '/' +expname
vis_dir = os.path.join(basedir, expname, f'compressed_struct_weights_{batch_size}')
os.makedirs(vis_dir, exist_ok=True)

n_struct_layers = args.mid_freeze_start
weight_matrix_width = [63, 256, 256, 256, 256, 256, 256]

a = torch.load(exp_folder+f'/{frame_a:06d}.tar')
a_fn = a['network_fn_state_dict']
a_fine = a['network_fine_state_dict']

# save struct layers
all_struct_layers = []
for i in range(batch_size):
    frame_cur = frame_a+i
    cur = torch.load(exp_folder + f'/{frame_cur:06d}.tar')
    cur_fn = cur['network_fn_state_dict']
    cur_fine = cur['network_fine_state_dict']
    cur_list = []
    for l in range(n_struct_layers):
        cur_list.append((cur_fn[f'pts_linears.{l}.weight']).cpu().numpy())
        cur_list.append((cur_fn[f'pts_linears.{l}.bias'])[:,None].cpu().numpy())
        cur_list.append((cur_fine[f'pts_linears.{l}.weight']).cpu().numpy())
        cur_list.append((cur_fine[f'pts_linears.{l}.bias'])[:,None].cpu().numpy())
    all_struct_layers.append(np.hstack(cur_list))
all_struct_np = np.stack(all_struct_layers)

# compressed_bytes = fpzip.compress(b_struct, precision=32, order='C')
compressed_bytes = fpzip.compress(all_struct_np, precision=16, order='C')
compressed_fn = vis_dir+f'/compressed_{frame_a:06d}-{frame_a+batch_size-1:06d}_c'
f = open(compressed_fn, 'wb')
f.write(compressed_bytes)
f.close()

# load struct layers
with open(compressed_fn, 'rb') as f:
    read_bytes = f.read()
all_struct_np = fpzip.decompress(read_bytes, order='C')

# compressed_fn = vis_dir+f'/compressed_{frame_a:06d}_c.npy'
# np.save(compressed_fn, a_struct)
# a_recovered = np.load(compressed_fn)

# create NeRF
for frame_i in range(frame_a, frame_a+batch_size):
    a_recovered = all_struct_np[0,frame_i-frame_a]
    args.no_reload = True  # don't auto-load when creating nerf
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, _ = create_nerf(args)
    w_start_column = 0
    for l in range(total_n_layers):
        if l < n_struct_layers:
            w_str = f'pts_linears.{l}.weight'
            b_str = f'pts_linears.{l}.bias'
            a_fn[w_str] = torch.Tensor(a_recovered[:,w_start_column:w_start_column+weight_matrix_width[l]]).to(device)
            a_fn[b_str] = torch.Tensor(a_recovered[:,w_start_column+weight_matrix_width[l]]).to(device)
            # # cur_weight = torch.Tensor(a_struct[:,w_start_column:w_start_column+weight_matrix_width[l]]).to(device)
            # # cur_bias = torch.Tensor(a_struct[:,w_start_column+weight_matrix_width[l]]).to(device)
            # render_kwargs_train['network_fn'].state_dict()[f'pts_linears.{l}.weight'].copy_(cur_weight)
            # render_kwargs_train['network_fn'].state_dict()[f'pts_linears.{l}.bias'].copy_(cur_bias)
            w_start_column += (weight_matrix_width[l] + 1)

            a_fine[w_str] = torch.Tensor(a_recovered[:,w_start_column:w_start_column+weight_matrix_width[l]]).to(device)
            a_fine[b_str] = torch.Tensor(a_recovered[:,w_start_column+weight_matrix_width[l]]).to(device)
            # # cur_weight = torch.Tensor(a_struct[:,w_start_column:w_start_column+weight_matrix_width[l]]).to(device)
            # # cur_bias = torch.Tensor(a_struct[:,w_start_column+weight_matrix_width[l]]).to(device)
            # render_kwargs_train['network_fine'].state_dict()[f'pts_linears.{l}.weight'].copy_(cur_weight)
            # render_kwargs_train['network_fine'].state_dict()[f'pts_linears.{l}.bias'].copy_(cur_bias)
            w_start_column += (weight_matrix_width[l]+1)

    render_kwargs_train['network_fn'].load_state_dict(a_fn)
    render_kwargs_train['network_fine'].load_state_dict(a_fine)
    render_kwargs_test['network_fn'].load_state_dict(a_fn)
    render_kwargs_test['network_fine'].load_state_dict(a_fine)

    ################################### load data ###############################
    if args.dataset_type == 'little_falls':
        data = load_LF_data(args.datadir, frame_i, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
    elif args.dataset_type == 'META':
        data = load_META_data(args.datadir, frame_i, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
    if data is None:
        raise "data is None!"
    else:
        images, poses, K, bds, render_poses, i_test, _ = data

    near = 0.
    far = 1.
    hwf = poses[0, :3, -1]
    poses = poses[:, :3, :4]
    print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
    if not isinstance(i_test, list):
        i_test = [i_test]
    if args.llffhold > 0:
        print('Auto LLFF holdout,', args.llffhold)
        # i_test = np.arange(images.shape[0])[::args.llffhold]
        i_test = np.array([args.llffhold])
    # i_test = [5,6,7,8,9]
    i_val = i_test
    i_train = np.array([i for i in np.arange(int(images.shape[0])) if
                        (i not in i_test and i not in i_val)])
    if args.no_ndc:
        near = np.ndarray.min(bds) * .9
        far = np.ndarray.max(bds) * 1.
    print('BOUNDS: NEAR FAR', near, far)
    # Cast intrinsics to right types
    H, W, focal = hwf
    H, W = int(H), int(W)
    hwf = [H, W, focal]
    bds_dict = {'near': near, 'far': far, }
    render_kwargs_train.update(bds_dict)
    render_kwargs_test.update(bds_dict)
    # Prepare raybatch tensor if batching random rays
    rays_rgb, permuted_i, _ = create_ray_rgb_K_batches(poses, H, W, K, images, i_train)
    rays_rgb = torch.Tensor(rays_rgb).to(device)
    images = torch.Tensor(images).to(device)
    poses = torch.Tensor(poses).to(device)
    render_poses = torch.Tensor(render_poses).to(device)
    print('done')
    i_batch = 0
    print('TRAIN views are', i_train)
    print('TEST views are', i_test)
    print('VAL views are', i_val)

    # render finetuned results
    img_i = i_val[0]
    psnr, rgb = render_from_view(img_i, render_kwargs_test, images, poses, K, H, W, args)
    filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{frame_i:04d}_{psnr.item():04f}.png')
    imageio.imwrite(filename, to8b(rgb.cpu().numpy()))

