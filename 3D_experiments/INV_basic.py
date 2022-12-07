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

# random middle layers actually also works
FREEZE_MIDDLE_LAYERS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False


def train():

    parser = config_parser()
    args = parser.parse_args()

    # Create log dir and copy the config file
    basedir = args.basedir
    expname = args.expname
    vis_dir = os.path.join(basedir, expname, 'nerf_esti')
    os.makedirs(os.path.join(basedir, expname), exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    # writer = SummaryWriter(vis_dir)
    f = os.path.join(basedir, expname, 'args.txt')
    with open(f, 'w') as file:
        for arg in sorted(vars(args)):
            attr = getattr(args, arg)
            file.write('{} = {}\n'.format(arg, attr))
    if args.config is not None:
        f = os.path.join(basedir, expname, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())
    do_stabilize = args.stablize_iter_prob > 0.

    # Create nerf model
    if args.is_nerf_baseline:
        args.no_reload = True
        args.no_skip_connect = False
        args.sample_dynamic_more = False
        args.freeze_start_frame = 10000
        args.stabilize_start_frame = 10000
    else:
        print("running INV")
        render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt_path = create_nerf(args)
        render_kwargs_train_prev, render_kwargs_test_prev, _, _, _, _ = create_nerf(args)
        global_step = start

        # load base model
        # ckpt_path = os.path.join(basedir, expname, '{:06d}.tar'.format(ckpt_frame))
        # print('Reloading from', ckpt_path)
        # ckpt = torch.load(ckpt_path)
        # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        # render_kwargs_train['network_fn'].load_state_dict(ckpt['network_fn_state_dict'])
        # if render_kwargs_train['network_fine'] is not None:
        #     render_kwargs_train['network_fine'].load_state_dict(ckpt['network_fine_state_dict'])
        if ckpt_path is not None:
            args.start_frame = int(os.path.basename(ckpt_path)[:6]) + 1

    N_iters = 200000 + 1
    # if args.is_nerf_baseline:
    #     args.i_weights = N_iters
    print('Begin')

    # Summary writers
    # writer = SummaryWriter(os.path.join(basedir, 'summaries', expname))

    near = max(0., args.near)
    far = 1.

    if args.use_huber:
        loss_fn = img2huber
    else:
        loss_fn = img2mse

    for f_i in range(100000):
        if args.is_nerf_baseline:
            print("running baseline NeRF without transfer")
            render_kwargs_train, render_kwargs_test, start, grad_vars, optimizer, ckpt_path = create_nerf(args)
            render_kwargs_train_prev, render_kwargs_test_prev, _, _, _, _ = create_nerf(args)
            global_step = start

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
        if masks is not None:
            print("loaded mask")
            masks = torch.Tensor(masks).to(device) > 0
            if args.sample_dynamic_more:
                dynamic_rays_rgb = rays_rgb[torch.logical_not(masks[:,0,0])]
                static_rays_rgb = rays_rgb[masks[:,0,0]]
                n_static_rays = int(args.sample_static_prob * N_rand) - 1
                n_dynamic_rays = N_rand - n_static_rays
                i_dynamic = 0
                i_static = 0

        print('done')
        i_batch = 0

        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        time_s0 = time.time()
        print(f"data loading: {time_s0-time_l0} sec / frame")
        print(f"stablize_iter_prob: {args.stablize_iter_prob}")

        # load previous model
        if cur_frame > 1 and cur_frame >= (args.stabilize_start_frame-1):
            prev_frame = cur_frame - 1
            prev_path = os.path.join(basedir, expname, '{:06d}.tar'.format(prev_frame))
            ckpt_prev = torch.load(prev_path)
            render_kwargs_train_prev['network_fn'].load_state_dict(ckpt_prev['network_fn_state_dict'])
            render_kwargs_train_prev['network_fine'].load_state_dict(ckpt_prev['network_fine_state_dict'])
            render_kwargs_test_prev['network_fn'].load_state_dict(ckpt_prev['network_fn_state_dict'])
            render_kwargs_test_prev['network_fine'].load_state_dict(ckpt_prev['network_fine_state_dict'])
            # if args.prev_fine_as_new_coarse:
            #     render_kwargs_train['network_fn'].load_state_dict(ckpt_prev['network_fine_state_dict'])
            #     render_kwargs_test['network_fn'].load_state_dict(ckpt_prev['network_fine_state_dict'])

        # freeze middle layers
        # if FREEZE_MIDDLE_LAYERS:
        #     for i, param in enumerate(render_kwargs_train['network_fine'].parameters()):
        #         if args.mid_freeze_start * 2 < i < args.mid_freeze_end * 2:
        #             param.requires_grad = False
        #     for i, param in enumerate(render_kwargs_train['network_fn'].parameters()):
        #         if args.mid_freeze_start * 2 < i < args.mid_freeze_end * 2:
        #             param.requires_grad = False

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

        time_store_prev = 0
        train_iter = args.i_weights if cur_frame >= args.freeze_start_frame else args.i_weights_warmup
        for i in trange(train_iter):
            if i % args.i_iter_img == 0 or i == train_iter-1:
                with torch.no_grad():
                    img_i = i_val[0]
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

            # NOTE: IMPORTANT!
            ###   update learning rate   ###
            if args.use_weight_decay:
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (i / decay_steps))
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lrate
            ################################

            #####  Core optimization loop  #####
            # Sample random ray batch from all images
            if args.sample_dynamic_more and masks is not None:
                batch = torch.cat([dynamic_rays_rgb[i_dynamic:i_dynamic+n_dynamic_rays],
                                  static_rays_rgb[i_static:i_static+n_static_rays]], dim=0)
                i_dynamic = (i_dynamic + n_dynamic_rays) % dynamic_rays_rgb.shape[0]
                i_static = (i_static + n_static_rays) % static_rays_rgb.shape[0]
            else:
                batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, K_rays = batch[:2], batch[2], batch[3:]

            indices_batch = indices[i_batch:i_batch + N_rand]
            if masks is not None:
                masks_cur = masks[i_batch:i_batch + N_rand,0,0]

            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                print("Shuffle data after an epoch!")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                indices = indices[rand_idx]
                if masks is not None:
                    masks = masks[rand_idx]
                    if args.sample_dynamic_more:
                        # should not happen when iter is low
                        dynamic_rays_rgb = rays_rgb[torch.logical_not(masks[:, 0, 0])]
                        static_rays_rgb = rays_rgb[masks[:, 0, 0]]
                        i_dynamic = 0
                        i_static = 0
                i_batch = 0

            rgb, disp, acc, extras = render(H, W, K_rays, chunk=args.chunk, rays=batch_rays,
                                            verbose=i < 10, retraw=True, **render_kwargs_train)

            optimizer.zero_grad()

            # img_loss = img2mse(rgb, target_s)
            img_loss = loss_fn(rgb, target_s)
            trans = [...,-1]
            loss = img_loss
            psnr = mse2psnr(img2mse(rgb, target_s))

            if 'rgb0' in extras:
                img_loss0 = loss_fn(extras['rgb0'], target_s)
                loss = loss + img_loss0

            #####  stabilize with previous frame  #####
            if masks is not None and cur_frame >= args.stabilize_start_frame and np.random.rand() < args.stablize_iter_prob:
                if masks_cur.any():
                    with torch.no_grad():
                        rgb_prev, disp_prev, acc_prev, extras_prev = render(H, W, K_rays, chunk=args.chunk, rays=batch_rays,
                                                                            verbose=i < 10, retraw=True, **render_kwargs_train_prev)
                    # img_loss_prev = img2mse(rgb[masks_cur], rgb_prev[masks_cur])
                    # stability_loss = img2mse(disp[masks_cur], disp_prev[masks_cur])
                    img_loss_prev = loss_fn(rgb[masks_cur], rgb_prev[masks_cur])
                    stability_loss = loss_fn(disp[masks_cur], disp_prev[masks_cur])
                    # if 'rgb0' in extras:
                    # img_loss_prev += img2mse(extras['rgb0'][masks_cur], extras_prev['rgb0'][masks_cur])
                    # stability_loss += img2mse(extras['disp0'][masks_cur], extras_prev['disp0'][masks_cur])
                    # img_loss_prev += loss_fn(extras['rgb0'][masks_cur], extras_prev['rgb0'][masks_cur])
                    # stability_loss += loss_fn(extras['disp0'][masks_cur], extras_prev['disp0'][masks_cur])

                    loss += (img_loss_prev + stability_loss / 10000) * 0.5
                    # psnr = mse2psnr(img_loss_prev)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(render_kwargs_train['network_fn'].parameters(), 2.)
            torch.nn.utils.clip_grad_norm_(render_kwargs_train['network_fine'].parameters(), 2.)
            optimizer.step()

            # for nerf baseline only
            if args.is_nerf_baseline and i % 10000 == 0:
                nerf_baseline_vis_dir = os.path.join(basedir, expname, 'nerf_esti')
                os.makedirs(nerf_baseline_vis_dir, exist_ok=True)
                for img_i in i_val:
                    # img_i = i_train[0]
                    gt_img = torch.Tensor(images[img_i]).to(device)
                    pose = poses[img_i, :3, :4]
                    with torch.no_grad():
                        rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                                        **render_kwargs_test)

                    psnr = mse2psnr(img2mse(rgb, gt_img))

                    filename = os.path.join(nerf_baseline_vis_dir, f'cam{img_i:02d}_iter{i:06d}_{psnr.item():06f}.png')
                    imageio.imwrite(filename, to8b(rgb.cpu().numpy()))

                path = os.path.join(basedir, expname, f'{cur_frame:06d}_iter{i:06d}.tar')
                torch.save({
                    'global_step': global_step,
                    'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
                    'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }, path)
                print('Saved checkpoints at', path)

        dt = time.time()-time0
        global_step += 1

        if i%args.i_print==0 and i > 0:
            tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():04f}  PSNR: {psnr.item():04f}")

        time_e0 = time.time()
        print(f"{time_e0-time_s0:.3f} sec / frame")

        # for img_i in i_val:
        img_i = i_val[0]
        gt_img = torch.Tensor(images[img_i]).to(device)
        pose = poses[img_i, :3, :4]

        # if cur_frame >= args.stabilize_start_frame:
        with torch.no_grad():
            rgb, disp, acc, extras = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                            **render_kwargs_test)
        psnr = mse2psnr(img2mse(rgb, gt_img))
        filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_{psnr.item():04f}.png')
        imageio.imwrite(filename, to8b(rgb.cpu().numpy()))

        # Rest is logging
        path = os.path.join(basedir, expname, '{:06d}.tar'.format(cur_frame))
        torch.save({
            'global_step': global_step,
            'network_fn_state_dict': render_kwargs_train['network_fn'].state_dict(),
            'network_fine_state_dict': render_kwargs_train['network_fine'].state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        print('Saved checkpoints at', path)
        ckpt_cur = torch.load(path)

        # Turn on testing mode
        if cur_frame % args.i_img == 0 and args.render_videos:
            testsavedir = os.path.join(basedir, expname, 'frame_{:04d}'.format(cur_frame))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                rgbs, disps = render_path(render_poses[:60:2], hwf, K[6], args.chunk, render_kwargs_test, savedir=testsavedir)
            print('Done, saving', rgbs.shape, disps.shape)

        global_step = 0


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
