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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


META, LF = ['META', 'little_falls']

# model weights for color layer, later finetuned on the frame based on the config file
start_frame = 1     # index starts at 1
continue_prev_exp = False
if continue_prev_exp:
    base_weight = '/playpen-ssd/mikewang/incremental_neural_videos/META_data/flame_salmon_1/down_2x/' \
                  'META_flame_salmon_3D_color_transfer_from_cut_roasted_beef/000001_iter004999.tar'
else:
    base_weight = None

def train():

    parser = config_parser()
    args = parser.parse_args()
    args.start_frame = start_frame

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    vis_dir = os.path.join(basedir, expname, 'nerf_esti')

    # create directories for logging
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

    # Create nerf model
    if args.is_nerf_baseline:
        print("\n#### 3D color transfer with NeRF (with skip connection) ####\n")
        args.no_reload = False              # need to load trained color layers
        if continue_prev_exp:
            args.ft_path = base_weight
        args.no_skip_connect = True         # orig nerf has skip connection
        args.freeze_start_frame = 0         # freeze all color layers

        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt_path = create_nerf(args)
        render_kwargs_train_prev, render_kwargs_test_prev, _, _, _, _ = create_nerf(args)
        if continue_prev_exp:
            global_step = start
            iter_start = start
        else:
            global_step = 0
            iter_start = 0

    else:
        print("3D color transfer with INV (with skip connection)")
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt_path = create_nerf(args)
        render_kwargs_train_prev, render_kwargs_test_prev, _, _, _, _ = create_nerf(args)
        global_step = start
        iter_start = 0

    print('Begin')

    near = max(0., args.near)
    far = 1.

    if args.use_huber:
        loss_fn = img2huber
    else:
        loss_fn = img2mse

    for f_i in range(100000):
        time_l0 = time.time()
        cur_frame = args.start_frame + f_i
        print(f'\n frame -- {cur_frame}')

        ### load little falls ###
        try:
            if args.dataset_type == 'little_falls':
                data = load_LF_data(args.datadir, cur_frame, args.factor, recenter=True, bd_factor=.75, spherify=args.spherify)
            elif args.dataset_type == 'META':
                data = load_META_data(args.datadir, cur_frame, args.factor, recenter=True, bd_factor=.75,
                                      spherify=args.spherify, load_masks=True)
            images, poses, K, bds, render_poses, i_test, masks = data
        except:
            print(f"failed to load frame {cur_frame}")
            continue

        hwf = poses[0, :3, -1]
        poses = poses[:, :3, :4]
        print('Loaded llff', images.shape, render_poses.shape, hwf, args.datadir)
        print('holdout test view:', args.llffhold)
        i_test =[args.llffhold]
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

        bds_dict = {
            'near': near,
            'far': far,
        }
        render_kwargs_train.update(bds_dict)
        render_kwargs_test.update(bds_dict)

        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        rays_rgb, indices, masks = create_ray_rgb_K_batches(poses, H, W, K, images, i_train, masks=masks)
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        render_poses = torch.Tensor(render_poses).to(device)
        indices = torch.Tensor(indices).to(device)
        i_batch = 0

        print('done')
        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        time_s0 = time.time()
        print(f"data loading: {time_s0-time_l0} sec / frame")

        # INV freezes color layers after a certain frame
        if cur_frame >= args.freeze_start_frame:
            for model_tmp in [render_kwargs_train['network_fine'], render_kwargs_train['network_fn']]:
                for l in range(len(model_tmp.pts_linears)):
                    if l < args.mid_freeze_start:
                        print(f"layer {l} not frozen")
                        continue
                    model_tmp.pts_linears[l].weight.requires_grad = False
                    model_tmp.pts_linears[l].bias.requires_grad = False
                model_tmp.views_linears[0].weight.requires_grad = False
                model_tmp.views_linears[0].bias.requires_grad = False
                model_tmp.feature_linear.weight.requires_grad = False
                model_tmp.feature_linear.bias.requires_grad = False
                model_tmp.alpha_linear.weight.requires_grad = False
                model_tmp.alpha_linear.bias.requires_grad = False
                model_tmp.rgb_linear.weight.requires_grad = False
                model_tmp.rgb_linear.bias.requires_grad = False

        train_iter = args.i_weights if cur_frame >= args.freeze_start_frame else args.i_weights_warmup
        for i in tqdm(range(iter_start, train_iter)):
            # nerf baseline: render intermediate results
            # if args.is_nerf_baseline and (i % args.i_iter_img == 0 or i == train_iter-1):
            if args.is_nerf_baseline and (i in [0, 500, 1000, 10000, 20000]):
                with torch.no_grad():
                    for img_i in [0, 1, 8, 17]:
                        gt_img = torch.Tensor(images[img_i]).to(device)
                        pose = poses[img_i, :3, :4]
                        rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)
                        psnr = mse2psnr(img2mse(rgb, gt_img))
                        filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_iter{i}_{psnr.item():04f}.png')
                        imageio.imwrite(filename, to8b(rgb.cpu().numpy()))

                    path = os.path.join(basedir, expname, f'{cur_frame:06d}_iter{i:06d}.tar')
                    torch.save({
                        'global_step': global_step,
                        'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                        'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                    }, path)
                    print('Saved checkpoints at', path)

            time0 = time.time()

            ###   update learning rate   ###
            if args.use_weight_decay:
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate

            # Sample random ray batch from all images
            batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, K_rays = batch[:2], batch[2], batch[3:]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                indices = indices[rand_idx]
                i_batch = 0

            rgb, disp, acc, extras = render(H, W, K_rays, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True, **render_kwargs_train)

            optimizer.zero_grad()

            img_loss = loss_fn(rgb, target_s)
            loss = img_loss
            psnr = mse2psnr(img2mse(rgb, target_s))

            if 'rgb0' in extras:
                img_loss0 = loss_fn(extras['rgb0'], target_s)
                loss = loss + img_loss0

            loss.backward()
            torch.nn.utils.clip_grad_norm_(render_kwargs_train['network_fn'].parameters(), 2.)
            torch.nn.utils.clip_grad_norm_(render_kwargs_train['network_fine'].parameters(), 2.)
            optimizer.step()

            dt = time.time()-time0
            global_step += 1

            if i%args.i_print==0 and i > 0:
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():04f}  PSNR: {psnr.item():04f}")

        time_e0 = time.time()
        print(f"{time_e0-time_s0:.3f} sec / frame")

        # render trained frame
        img_i = i_val[0]
        gt_img = torch.Tensor(images[img_i]).to(device)
        pose = poses[img_i, :3, :4]
        with torch.no_grad():
            rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                            **render_kwargs_test)
        psnr = mse2psnr(img2mse(rgb, gt_img))
        filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_{psnr.item():04f}.png')
        imageio.imwrite(filename, to8b(rgb.cpu().numpy()))

        # save weights
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(cur_frame))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
            'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)

        # Turn on testing mode
        if cur_frame % args.i_img == 0 and args.render_videos:
            testsavedir = os.path.join(basedir, expname, 'frame_{:04d}'.format(cur_frame))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                rgbs, disps = render_path(render_poses[:60:2], hwf, K[6], args.chunk, render_kwargs_test, savedir=testsavedir)
            print('Done, saving', rgbs.shape, disps.shape)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
