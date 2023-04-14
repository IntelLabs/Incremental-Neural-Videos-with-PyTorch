import os, sys
import numpy as np
import imageio
import cv2
import time
from tqdm import tqdm, trange
import torch

import matplotlib.pyplot as plt

from config_parser import config_parser
from model_helpers import *
from helpers import create_nerf, render, render_blend_two_models, render_path, create_ray_rgb_K_batches
from load_llff import load_LF_data, load_META_data

START_FRAME = 1
N_FRAMES=300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

def npar_empty() :
    empty = {"render_kwargs_train": None,
             "render_kwargs_test": None,
             "start": None,
             "grad_vars":None,
             "optimizer":None,
             "ckpt_path":None
             }
    return empty

def npar_create_nerf(args):
    render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt_path = create_nerf(args)
    return {"render_kwargs_train": render_kwargs_train,
             "render_kwargs_test": render_kwargs_test,
             "start": start,
             "grad_vars":grad_vars,
             "optimizer":optimizer,
             "ckpt_path":ckpt_path
             }

def train():

    parser = config_parser()
    args = parser.parse_args()

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    vis_dir = os.path.join(basedir, expname, '3Dvideo')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)

    args.start_frame = START_FRAME

    print('Begin')

    near = max(0., args.near)
    far = 1.

    if args.dataset_type == 'little_falls':
        data = load_LF_data(args.datadir, args.start_frame, args.factor, recenter=True, bd_factor=.75,
                            spherify=args.spherify)
    elif args.dataset_type == 'META':
        data = load_META_data(args.datadir, args.start_frame, args.factor, recenter=True, bd_factor=.75,
                              spherify=args.spherify, load_masks=True)
    images, poses, K, bds, render_poses, i_test, masks = data

    for f_i in range(N_FRAMES):
        cur_frame = args.start_frame + f_i
        print("running INV")

        args.ft_path = os.path.join(basedir, args.expname, f'{cur_frame:06d}_static.tar')

        nerf = [npar_create_nerf(args)]
        args.expname = expname + '_dynamic'  # override checkpoint directory temporarily
        args.ft_path = os.path.join(basedir, args.expname, f'{cur_frame:06d}_dynamic.tar')
        nerf.append(npar_create_nerf(args))

        args.expname = expname
        render_kwargs_train_two_models = {}
        for key in nerf[0]['render_kwargs_train']:
            if key not in ['network_fine', 'network_fn']:
                render_kwargs_train_two_models[key] = nerf[0]['render_kwargs_train'][key]
        render_kwargs_train_two_models['network_fn_s'] = nerf[0]['render_kwargs_train']['network_fn']
        render_kwargs_train_two_models['network_fine_s'] = nerf[0]['render_kwargs_train']['network_fine']
        render_kwargs_train_two_models['network_fn_d'] = nerf[1]['render_kwargs_train']['network_fn']
        render_kwargs_train_two_models['network_fine_d'] = nerf[1]['render_kwargs_train']['network_fine']
        render_kwargs_train_two_models['offset_d'] = 0
        render_kwargs_train_two_models['scale_d'] = 1
        render_kwargs_train_two_models['mean_d'] = 0

        render_kwargs_test_two_models = {k: render_kwargs_train_two_models[k] for k in render_kwargs_train_two_models}
        render_kwargs_test_two_models['perturb'] = False
        render_kwargs_test_two_models['raw_noise_std'] = 0.



        time_l0 = time.time()
        print(f'\n frame -- {cur_frame}')

        hwf = poses[0, :3, -1]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        if args.no_ndc:
            near = np.ndarray.min(bds) * .9
            far = np.ndarray.max(bds) * 1.
        print('BOUNDS: NEAR FAR', near, far)

        # Cast intrinsics to right types
        H, W, focal = hwf
        H, W = int(H), int(W)
        hwf = [H, W, focal]

        render_poses = torch.Tensor(render_poses).to(device)

        print('done')

        time_s0 = time.time()
        print(f"data loading: {time_s0-time_l0} sec / frame")

        # if cur_frame >= args.stabilize_start_frame:
        with torch.no_grad():
            rgb, disp, acc, extras = render_blend_two_models(H, W, K[6], chunk=args.chunk,
                                                             c2w=render_poses[(cur_frame)%120, :3,:4],
                                                             **render_kwargs_test_two_models)
        filename = os.path.join(vis_dir, f'{cur_frame:06d}.png')
        imageio.imwrite(filename, to8b(rgb.cpu().numpy()))


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
