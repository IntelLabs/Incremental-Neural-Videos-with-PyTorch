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

import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

from config_parser import config_parser
from model_helpers import *
from helpers import create_nerf, render, render_path, create_ray_rgb_K_batches
from load_llff import load_llff_data, load_LF_data, load_META_data

META, LF = ['META', 'little_falls']

base_frame = 462
first_frame_to_process = 470 # base_frame+1
DO_SWAP_LAYERS = True

train_first_n_layers = 0
train_last_layer = False
swap_n_layer = 11   #nerf has 8 + 1 alpha & 3 color
# swap_n_layer = 4
DO_SWAP_ALPHA = False
DO_FINETUNE_SWAPPED = False
DO_INCRE_XFER_EARLY_LAYERS = False
DO_SWAP_AFTER_FINETUNE_AS_WELL = False
n_frames = 10000

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def swap_analyze_layers(cur_frame, l_list, weight_cur, weight_fine_cur,
                        render_kwargs_train, render_kwargs_test, i_val,
                        images, poses, H, W, K, args, save_image_dir, model_dir,
                        is_refined, do_render):
    version = 'refined' if is_refined else 'raw_swap'
    for l in l_list:
        if l <= 7:
            w_str = f'pts_linears.{l}.weight'
            b_str = f'pts_linears.{l}.bias'
        elif l == 8:
            w_str = f'feature_linear.weight'
            b_str = f'feature_linear.bias'
        elif l == 9:
            w_str = f'views_linears.{0}.weight'
            b_str = f'views_linears.{0}.bias'
        elif l == 10:
            w_str = f'rgb_linear.weight'
            b_str = f'rgb_linear.weight'
        if w_str in weight_cur:
            render_kwargs_train['network_fn'].state_dict()[w_str].copy_(weight_cur[w_str])
            render_kwargs_train['network_fn'].state_dict()[b_str].copy_(weight_cur[b_str])
            render_kwargs_train['network_fine'].state_dict()[w_str].copy_(weight_fine_cur[w_str])
            render_kwargs_train['network_fine'].state_dict()[b_str].copy_(weight_fine_cur[b_str])

        if do_render:
            for img_i in i_val:
                # img_i = i_train[0]
                gt_img = torch.Tensor(images[img_i]).to(device)
                pose = poses[img_i, :3, :4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose, **render_kwargs_test)
                psnr = mse2psnr(img2mse(rgb, gt_img))
                filename = os.path.join(save_image_dir, f'cam{img_i:02d}_frame{base_frame:04d}_e{cur_frame:04d}_'
                                                  f'{version}_{l_list[0]}-{l_list[l]}_{psnr.item():05f}.png')
                imageio.imwrite(filename, to8b(rgb.cpu().numpy()))

    if DO_SWAP_ALPHA:
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, _ = create_nerf(args)
        ckpt_path = os.path.join(model_dir, f'{base_frame:06d}.tar')
        print('Reloading from', ckpt_path)
        ckpt = torch.load(ckpt_path)
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        render_kwargs_train['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
        if render_kwargs_train['network_fine'] is not None:
            render_kwargs_train['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])
        for l in list(range(7 + 1 + 1)):
            if l <= 7:
                w_str = f'pts_linears.{l}.weight'
                b_str = f'pts_linears.{l}.bias'
            else:
                w_str = f'alpha_linear.weight'
                b_str = f'alpha_linear.weight'
            render_kwargs_train['network_fn'].state_dict()[w_str].copy_(weight_cur[w_str])
            render_kwargs_train['network_fn'].state_dict()[b_str].copy_(weight_cur[b_str])
            render_kwargs_train['network_fine'].state_dict()[w_str].copy_(weight_fine_cur[w_str])
            render_kwargs_train['network_fine'].state_dict()[b_str].copy_(weight_fine_cur[b_str])

        if do_render:
            for img_i in i_val:
                # img_i = i_train[0]
                gt_img = torch.Tensor(images[img_i]).to(device)
                pose = poses[img_i, :3, :4]
                with torch.no_grad():
                    rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose, **render_kwargs_test)

                psnr = mse2psnr(img2mse(rgb, gt_img))
                filename = os.path.join(save_image_dir, f'cam{img_i:02d}_frame{base_frame:04d}_e{cur_frame:04d}_'
                                                  f'{version}_{l_list[0]}-7_alpha_{psnr.item():05f}.png')
                imageio.imwrite(filename, to8b(rgb.cpu().numpy()))


def finetune_model(i_batch, rays_rgb, optimizer, render_kwargs_train, H, W, args):
    N_rand = args.N_rand

    global_step = 0
    for i in trange(args.i_weights):
        time0 = time.time()
        # Sample random ray batch from all images
        batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
        batch = torch.transpose(batch, 0, 1)
        batch_rays, target_s, K_rays = batch[:2], batch[2], batch[3:]
        i_batch += N_rand
        if i_batch >= rays_rgb.shape[0]:
            print("Shuffle data after an epoch!")
            rand_idx = torch.randperm(rays_rgb.shape[0])
            rays_rgb = rays_rgb[rand_idx]
            i_batch = 0
        #####  Core optimization loop  #####
        rgb, disp, acc, extras = render(H, W, K_rays, chunk=args.chunk, rays=batch_rays, verbose=i < 10, retraw=True,
                                        **render_kwargs_train)
        optimizer.zero_grad()
        img_loss = img2mse(rgb, target_s)
        trans = extras['raw'][..., -1]
        loss = img_loss
        psnr = mse2psnr(img_loss)
        if 'rgb0' in extras:
            img_loss0 = img2mse(extras['rgb0'], target_s)
            loss = loss + img_loss0
            psnr0 = mse2psnr(img_loss0)
        loss.backward()
        optimizer.step()
        # NOTE: IMPORTANT!
        ###   update learning rate   ###
        decay_rate = 0.1
        decay_steps = args.lrate_decay * 1000
        new_lrate = args.lrate * (decay_rate ** (global_step / decay_steps))
        for param_group in optimizer.param_groups:
            param_group['lr'] = new_lrate
        dt = time.time() - time0
        global_step += 1
        if i % args.i_print == 0 and i > 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():04f}  PSNR: {psnr.item():04f}")


def render_from_view(img_i, render_kwargs_test, images, poses, K, H, W, args):
    # img_i = i_train[0]
    gt_img = torch.Tensor(images[img_i]).to(device)
    pose = poses[img_i, :3, :4]
    with torch.no_grad():
        rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                        **render_kwargs_test)

    psnr = mse2psnr(img2mse(rgb, gt_img))
    return psnr, rgb


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    # vis_dir = os.path.join(basedir, expname, 'nerf_esti')
    swap_dir = os.path.join(basedir, expname, 'swap')
    os.makedirs(swap_dir, exist_ok=True)
    incre_xfer_dir = os.path.join(basedir, expname, f'{train_first_n_layers}{"&last_" if train_last_layer else ""}'
                                                        f'layer_incre_xfer')
    os.makedirs(incre_xfer_dir, exist_ok=True)
    model_dir = os.path.join(basedir, expname)

    N_iters = 200000 + 1
    print('Begin')

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    # initial values
    near = 0.
    far = 1.
    
    render_kwargs_train = {}
    render_kwargs_test = {}
    
    for cur_frame in range(first_frame_to_process, args.start_frame+n_frames):

        # read cur frame
        if DO_SWAP_LAYERS:
            try:
                ckpt_path_cur = os.path.join(model_dir, f'{cur_frame:06d}.tar')
                ckpt_cur = torch.load(ckpt_path_cur)
                weight_cur = ckpt_cur['network_fn_state_dict']
                weight_fine_cur = ckpt_cur['network_fine_state_dict']
            except:
                print(f"failed to load {cur_frame:06d}.tar")
                continue

        # load base frame
        args.no_reload = True  # don't auto-load when creating nerf
        if DO_SWAP_LAYERS or cur_frame == first_frame_to_process:
            render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, _ = create_nerf(args)
            ckpt_path = os.path.join(model_dir, f'{base_frame:06d}.tar')
            print('Reloading from', ckpt_path)
            ckpt = torch.load(ckpt_path)
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            render_kwargs_train['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
            if render_kwargs_train['network_fine'] is not None:
                render_kwargs_train['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])

        ################################### load data ###############################
        if args.dataset_type == LF:
            data = load_LF_data(args.datadir, cur_frame, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
        elif args.dataset_type == META:
            data = load_META_data(args.datadir, cur_frame, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
        if data is None:
            continue
        else:
            images, poses, K, bds, render_poses, i_test, _ = data

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
        bds_dict = { 'near': near, 'far': far, }
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

        ###########                        SWAP LAYERS & RENDER                         ##########
        # swap weight
        if DO_SWAP_LAYERS:
            l_list = list(range(swap_n_layer))
            swap_analyze_layers(cur_frame, l_list, weight_cur, weight_fine_cur,
                                render_kwargs_train, render_kwargs_test, i_val,
                                images, poses, H, W, K, args, swap_dir, model_dir, is_refined=False, do_render=True)

            l_list = list(range(train_first_n_layers))
            if DO_FINETUNE_SWAPPED:
                # fix later layers
                for i, param in enumerate(render_kwargs_train['network_fine'].parameters()):
                    if i < train_first_n_layers*2:
                        continue
                    param.requires_grad = False
                for i, param in enumerate(render_kwargs_train['network_fn'].parameters()):
                    if i < train_first_n_layers*2:
                        continue
                    param.requires_grad = False

                finetune_model(i_batch, rays_rgb, optimizer, render_kwargs_train, H, W, args)

                # save structure layer weights
                structure_ckpt_path = os.path.join(basedir, expname, f'refined_swap_base{base_frame:04d}_cur{cur_frame:04d}'
                                                                     f'_train_{train_first_n_layers}_layers.tar')
                structure_layer_state_dict = {'network_fn': {}, 'network_fine': {}}
                for l in l_list:
                    structure_layer_state_dict['network_fn'][f'pts_linears.{l}.weight'] \
                        = render_kwargs_train['network_fn'].state_dict()[f'pts_linears.{l}.weight']
                    structure_layer_state_dict['network_fn'][f'pts_linears.{l}.bias'] \
                        = render_kwargs_train['network_fn'].state_dict()[f'pts_linears.{l}.bias']
                    structure_layer_state_dict['network_fine'][f'pts_linears.{l}.weight'] \
                        = render_kwargs_train['network_fine'].state_dict()[f'pts_linears.{l}.weight']
                    structure_layer_state_dict['network_fine'][f'pts_linears.{l}.bias'] \
                        = render_kwargs_train['network_fine'].state_dict()[f'pts_linears.{l}.bias']
                torch.save(structure_layer_state_dict, structure_ckpt_path)

            # render finetuned results
            for img_i in i_val:
                psnr, rgb = render_from_view(img_i, render_kwargs_test, images, poses, K, H, W, args)
                filename = os.path.join(swap_dir, f'cam{img_i:02d}_frame{base_frame:04d}_e{cur_frame:04d}'
                                                  f'_{"refined_swap" if DO_FINETUNE_SWAPPED else "raw"}_'
                                                  f'_{psnr.item():04f}.png')
                imageio.imwrite(filename, to8b(rgb.cpu().numpy()))


        #######              INCREMENTAL TRANSFER ON EARLY LAYERS W/O SWAP            #########
        if DO_INCRE_XFER_EARLY_LAYERS:
            if DO_SWAP_LAYERS:
                # get fresh base model, avoid bug in usin the swapped model
                render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, _ = create_nerf(args)
                ckpt_path = os.path.join(model_dir, f'{base_frame:06d}.tar')
                print('Reloading from', ckpt_path)
                ckpt = torch.load(ckpt_path)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                render_kwargs_train['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
                if render_kwargs_train['network_fine'] is not None:
                    render_kwargs_train['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])

            # fix later layers
            for i, param in enumerate(render_kwargs_train['network_fine'].parameters()):
                if i < train_first_n_layers*2:
                    continue
                param.requires_grad = False
            for i, param in enumerate(render_kwargs_train['network_fn'].parameters()):
                if i < train_first_n_layers*2:
                    continue
                param.requires_grad = False
            if train_last_layer:
                render_kwargs_train['network_fine'].alpha_linear.weight.requires_grad = True
                render_kwargs_train['network_fine'].alpha_linear.bias.requires_grad = True
                render_kwargs_train['network_fine'].rgb_linear.weight.requires_grad = True
                render_kwargs_train['network_fine'].rgb_linear.bias.requires_grad = True

                render_kwargs_train['network_fn'].alpha_linear.weight.requires_grad = True
                render_kwargs_train['network_fn'].alpha_linear.bias.requires_grad = True
                render_kwargs_train['network_fn'].rgb_linear.weight.requires_grad = True
                render_kwargs_train['network_fn'].rgb_linear.bias.requires_grad = True

            finetune_model(i_batch, rays_rgb, optimizer, render_kwargs_train, H, W, args)

            # render
            for img_i in i_val:
                psnr, rgb = render_from_view(img_i, render_kwargs_test, images, poses, K, H, W, args)
                filename = os.path.join(incre_xfer_dir, f'cam{img_i:02d}_frame{base_frame:04d}_e{cur_frame:04d}'
                                                  f'_rincre_xfer_{psnr.item():04f}.png')
                imageio.imwrite(filename, to8b(rgb.cpu().numpy()))

            # save trained weights
            structure_ckpt_path = os.path.join(basedir, expname, f'incre_xfer_base{base_frame:04d}_cur{cur_frame:04d}'
                                                                 f'_train_{train_first_n_layers}'
                                                                 f'{"&last" if train_last_layer else ""}_layers.tar')
            structure_layer_state_dict = {'network_fn': {}, 'network_fine': {}}
            l_list = list(range(train_first_n_layers))
            for l in l_list:
                structure_layer_state_dict['network_fn'][f'pts_linears.{l}.weight'] \
                    = render_kwargs_train['network_fn'].state_dict()[f'pts_linears.{l}.weight']
                structure_layer_state_dict['network_fn'][f'pts_linears.{l}.bias'] \
                    = render_kwargs_train['network_fn'].state_dict()[f'pts_linears.{l}.bias']
                structure_layer_state_dict['network_fine'][f'pts_linears.{l}.weight'] \
                    = render_kwargs_train['network_fine'].state_dict()[f'pts_linears.{l}.weight']
                structure_layer_state_dict['network_fine'][f'pts_linears.{l}.bias'] \
                    = render_kwargs_train['network_fine'].state_dict()[f'pts_linears.{l}.bias']

                if train_last_layer:
                    structure_layer_state_dict['network_fine']['alpha_linear'] \
                        = render_kwargs_train['network_fine'].state_dict()[f'alpha_linear.weight']
                    structure_layer_state_dict['network_fine']['alpha_linear'] \
                        = render_kwargs_train['network_fine'].state_dict()[f'alpha_linear.bias']
                    structure_layer_state_dict['network_fine']['rgb_linear'] \
                        = render_kwargs_train['network_fine'].state_dict()[f'rgb_linear.weight']
                    structure_layer_state_dict['network_fine']['rgb_linear'] \
                        = render_kwargs_train['network_fine'].state_dict()[f'rgb_linear.bias']

                    structure_layer_state_dict['network_fn']['alpha_linear'] \
                        = render_kwargs_train['network_fn'].state_dict()[f'alpha_linear.weight']
                    structure_layer_state_dict['network_fn']['alpha_linear'] \
                        = render_kwargs_train['network_fn'].state_dict()[f'alpha_linear.bias']
                    structure_layer_state_dict['network_fn']['rgb_linear'] \
                        = render_kwargs_train['network_fn'].state_dict()[f'rgb_linear.weight']
                    structure_layer_state_dict['network_fn']['rgb_linear'] \
                        = render_kwargs_train['network_fn'].state_dict()[f'rgb_linear.bias']

            torch.save(structure_layer_state_dict, structure_ckpt_path)

            # swap weight to see if freezing later layers cause larger motion in earlier layers
            if DO_SWAP_AFTER_FINETUNE_AS_WELL:
                # load structure layers
                finetuned_ckpt_cur = torch.load(structure_ckpt_path)
                weight_cur = finetuned_ckpt_cur['network_fn']
                weight_fine_cur = finetuned_ckpt_cur['network_fine']

                # load base model
                render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, _ = create_nerf(args)
                ckpt_path = os.path.join(model_dir, f'{base_frame:06d}.tar')
                print('Reloading from', ckpt_path)
                ckpt = torch.load(ckpt_path)
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                render_kwargs_train['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
                if render_kwargs_train['network_fine'] is not None:
                    render_kwargs_train['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])

                l_list = list(range(swap_n_layer))
                swap_analyze_layers(cur_frame, l_list, weight_cur, weight_fine_cur,
                                    render_kwargs_train, render_kwargs_test, i_val,
                                    images, poses, H, W, K, args, incre_xfer_dir, model_dir,
                                    is_refined=True, do_render=True)

        if False:
            testsavedir = os.path.join(basedir, expname, 'frame_{:04d}'.format(cur_frame))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                rgbs, disps = render_path(render_poses[:60], hwf, K[6], args.chunk, render_kwargs_test,
                                          savedir=testsavedir)
            print('Done, saving', rgbs.shape, disps.shape)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
