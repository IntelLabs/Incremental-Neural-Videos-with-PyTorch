import os, sys
import numpy as np
import imageio
import cv2
import time
from tqdm import tqdm, trange

import matplotlib.pyplot as plt

from config_parser import config_parser
from model_helpers import *
from helpers import create_nerf, render, render_path, create_ray_rgb_K_batches
from load_llff import load_LF_data, load_META_data


START_FRAME = 470
N_FRAMES = 300

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


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

        # Create nerf model
        print("running INV")
        args.ft_path = os.path.join(basedir, expname, f'{cur_frame:06d}.tar')
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt_path = create_nerf(args)
        render_kwargs_train_prev, render_kwargs_test_prev, _, _, _, _ = create_nerf(args)

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

        # Turn on testing mode
        with torch.no_grad():
            rgbs, disps = render_path(render_poses[(cur_frame)%120,None], hwf, K[6], args.chunk, render_kwargs_test,
                                      savedir=vis_dir, save_name=f'{cur_frame:06d}')
        print('Done, saving', rgbs.shape, disps.shape)



if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
