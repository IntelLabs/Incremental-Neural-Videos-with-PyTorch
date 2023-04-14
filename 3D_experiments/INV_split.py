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

# random middle layers actually also works
FREEZE_MIDDLE_LAYERS = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
np.random.seed(0)
DEBUG = False

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
    print("running split INV")
    parser = config_parser()
    args = parser.parse_args()
    basedir = args.basedir
    expname = args.expname

    # Create static & dynamic nerf model
    nerf = []
    for i, postfix in enumerate(['_static', '_dynamic']):
        if args.pretraining_static and i>0:
            break
        args.expname = expname + postfix
        exp_root_dir = os.path.join(basedir, args.expname)
        vis_dir = os.path.join(basedir, args.expname, 'nerf_esti')
        os.makedirs(exp_root_dir, exist_ok=True)
        os.makedirs(vis_dir, exist_ok=True)

        f = os.path.join(exp_root_dir, 'args.txt')
        with open(f, 'w') as file:
            for arg in sorted(vars(args)):
                attr = getattr(args, arg)
                file.write('{} = {}\n'.format(arg, attr))
        f = os.path.join(exp_root_dir, 'config.txt')
        with open(f, 'w') as file:
            file.write(open(args.config, 'r').read())

        os.makedirs(os.path.join(basedir, args.expname), exist_ok=True)
        nerf.append(npar_create_nerf(args))
    args.expname = expname

    # create training context
    render_kwargs_train_two_models = {}
    for key in nerf[0]['render_kwargs_train']:  # copy all the settings
        if key not in ['network_fine', 'network_fn']:
            render_kwargs_train_two_models[key] = nerf[0]['render_kwargs_train'][key]
    for i in range(len(nerf)):
        postfix = ['_s', '_d'][i]
        render_kwargs_train_two_models['network_fn'+f'{postfix}'] = nerf[i]['render_kwargs_train']['network_fn']
        render_kwargs_train_two_models['network_fine'+f'{postfix}'] = nerf[i]['render_kwargs_train']['network_fine']

    # create testing context
    render_kwargs_test_two_models = {k: render_kwargs_train_two_models[k] for k in render_kwargs_train_two_models}
    render_kwargs_test_two_models['perturb'] = False
    render_kwargs_test_two_models['raw_noise_std'] = 0.

    if args.pretraining_static:
        global_step = nerf[0]["start"]
        # load base model
        if nerf[0]["ckpt_path"] is not None:
            args.start_frame = int(os.path.basename(nerf[0]["ckpt_path"])[:6]) + 1
        args.freeze_start_frame = 100000

        print("\n\n##### pretraining static NeRF #####")
    else:
        global_step = nerf[-1]["start"]
        if nerf[-1]["ckpt_path"] is not None:
            args.start_frame = int(os.path.basename(nerf[-1]["ckpt_path"])[:6]) + 1

        print("\n\n##### train dynamic INV #####")

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
            data = load_META_data(args.datadir, cur_frame, args.factor, recenter=True, bd_factor=.75,
                                  spherify=args.spherify, load_masks=True)
            images, poses, K, bds, render_poses, i_test, masks0 = data
        except:
            print(f"failed to load frame {cur_frame}")
            continue

        if masks0 is None:
            print("no mask, skipping this frame")
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
        for nr in nerf:
            nr["render_kwargs_train"].update(bds_dict)
            nr["render_kwargs_test"].update(bds_dict)

        # Prepare raybatch tensor if batching random rays
        N_rand = args.N_rand
        rays_rgb, indices, masks = create_ray_rgb_K_batches(poses, H, W, K, images, i_train, masks=masks0)
        rays_rgb = torch.Tensor(rays_rgb).to(device)
        images = torch.Tensor(images).to(device)
        poses = torch.Tensor(poses).to(device)
        render_poses = torch.Tensor(render_poses).to(device)
        indices = torch.Tensor(indices).to(device)

        i_batch = 0
        print("loaded mask")
        masks = torch.Tensor(masks).to(device) > 0
        dynamic_rays_rgb = rays_rgb[torch.logical_not(masks[:,0,0])]
        static_rays_rgb = rays_rgb[masks[:,0,0]]
        i_dynamic = 0
        i_static = 0

        print('TRAIN views are', i_train)
        print('TEST views are', i_test)
        print('VAL views are', i_val)

        time_s0 = time.time()
        print(f"data loading: {time_s0-time_l0} sec / frame")
        print(f"stablize_iter_prob: {args.stablize_iter_prob}")

        # Note: training of a video is staged:
        # Stage 0: train static background on a bunch of frames
        # Stage 1: quality not good yet
        #           1.1: 50% fgd nerf masked, 1.2: 25%, fgd nerf not masked, 25% both fgd & bgd
        # Stage 2: quality good
        #           1.1: 75% fgd nerf masked, 1.2: 25%, fgd nerf not masked, bgd fixed
        # Stage 3: quality good & start INV (freeze later layers)
        train_iter = args.i_weights if cur_frame >= args.freeze_start_frame else args.i_weights_warmup
        if cur_frame >= 9:
            quality_good = True
            train_unmasked_iter = train_iter * 0.75 // 1
            train_both_iter = train_iter * 0.75 // 1 * 1000  # means never train bgd cuz its well trained
        else:
            quality_good = False
            train_unmasked_iter = train_iter * 0.5 // 1
            train_both_iter = train_iter * 0.75 // 1

        # freezing layers when quality is stable:
        # 1. static NeRF: freeze entire model
        # 2. dynamic INV:   if not at Stage 3 (INV) yet, don't freeze
        #                   if at Stage 3, freeze color layers
        do_freeze = cur_frame >= args.freeze_start_frame
        if (do_freeze or quality_good) and not args.pretraining_static:
            for i in range(len(nerf)):
                if i == 0:
                    freeze_start_layer = 0
                else:
                    if not do_freeze:
                        continue
                    else:
                        freeze_start_layer = args.mid_freeze_start

                nr = nerf[i]
                models = [nr["render_kwargs_train"]['network_fine'], nr["render_kwargs_train"]['network_fn']]

                for model_tmp in models:
                    for l in range(len(model_tmp.pts_linears)):
                        if l < freeze_start_layer:
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

        # static and dynamic NeRF are trained separately
        if args.pretraining_static:
            ni = 0
        else:
            ni = 1

        nr = nerf[ni]
        for i in trange(train_iter):
            time0 = time.time()
            cur_render_kwargs_train = nr['render_kwargs_train']
            cur_opti = nr['optimizer']

            ###   update learning rate   ###
            if args.use_weight_decay:
                decay_rate = 0.1
                decay_steps = args.lrate_decay * 1000
                new_lrate = args.lrate * (decay_rate ** (i / decay_steps))

                for param_group in nr["optimizer"].param_groups:
                    param_group['lr'] = new_lrate

            if ni == 0:
                # Note: Stage 0: pretraining static background NeRF

                batch = static_rays_rgb[i_static:i_static + N_rand]
                i_static = (i_static + N_rand) % static_rays_rgb.shape[0]
                do_train_unmasked = False
                do_train_both = False
                do_enforce_zero_rgba = False
            else:
                # Note: Stage 1-3: training dynamic foreground INV

                if i < train_unmasked_iter:
                    # First part of training: use masked fgd images
                    batch = dynamic_rays_rgb[i_dynamic:i_dynamic + N_rand]
                    i_dynamic = (i_dynamic + N_rand) % dynamic_rays_rgb.shape[0]
                    do_train_unmasked = False
                    do_train_both = False
                    do_enforce_zero_rgba = False
                else:
                    do_static = (not quality_good) and (np.random.rand() > 0.5)
                    # do_static = False
                    if do_static:
                        # there's a 50% chance to train static bgd NeRF
                        batch = static_rays_rgb[i_static:i_static + N_rand]
                        i_static = (i_static + N_rand) % static_rays_rgb.shape[0]
                        cur_render_kwargs_train = nerf[0]['render_kwargs_train']
                        cur_opti = nerf[0]['optimizer']

                        do_train_unmasked = False
                        do_train_both = False
                        do_enforce_zero_rgba = False
                    else:
                        # train INV with entire unmasked images
                        batch = rays_rgb[i_batch:i_batch + N_rand]  # [B, 2+1, 3*?]
                        do_train_unmasked = True
                        zero_alpha_weight = 0.01
                        do_enforce_zero_rgba = args.enforce_zero_rgba
                        # if quality_good:
                        #     do_enforce_zero_rgba = args.enforce_zero_rgba
                        #     zero_alpha_weight = 0.01
                        # else:
                        #     do_enforce_zero_rgba = False
                        #     zero_alpha_weight = 0.

                        # for final iters, train fgd INV & bgd NeRF together
                        if i < train_both_iter:
                            do_train_both = False
                        else:
                            do_train_both = True

            if batch.shape[0] <= 0:
                print('Zero batch:', i)
                continue
            batch = torch.transpose(batch, 0, 1)
            batch_rays, target_s, K_rays = batch[:2], batch[2], batch[3:]

            # for dynamic NeRF, render static background NeRF
            if args.pretraining_static or not do_train_unmasked:
                rgb, disp, acc, extras = render(H, W, K_rays, chunk=args.chunk, rays=batch_rays,
                                                verbose=i < 10, retraw=True, **nr["render_kwargs_train"])
            else:
                rgb, disp, acc, extras = render_blend_two_models(H, W, K_rays, chunk=args.chunk, rays=batch_rays,
                                                                 verbose=i<10, retraw=True, **render_kwargs_train_two_models)

            # backprop
            cur_opti.zero_grad()
            if do_train_both:
                nerf[0]['optimizer'].zero_grad()

            img_loss = loss_fn(rgb, target_s)
            loss = img_loss
            if 'rgb0' in extras:
                rgb0 = extras['rgb0']
                img_loss0 = loss_fn(rgb0, target_s)
                loss = loss + img_loss0

            if do_enforce_zero_rgba:
                loss = loss + zero_alpha_weight * (
                        (extras['rgb_d'] + extras['rgb0_d'])[masks[i_batch:i_batch + N_rand, 0, 0]].mean()
                        + 0.3 * (extras['acc_map_d'] + extras['acc_map0_d'])[masks[i_batch:i_batch + N_rand, 0, 0]].mean())

            loss.backward()
            torch.nn.utils.clip_grad_norm_(cur_render_kwargs_train['network_fn'].parameters(), 2.)
            torch.nn.utils.clip_grad_norm_(cur_render_kwargs_train['network_fine'].parameters(), 2.)

            cur_opti.step()
            if do_train_both:
                nerf[0]['optimizer'].step()

            # reshuffle if needed
            i_batch += N_rand
            if i_batch >= rays_rgb.shape[0]:
                # should not happen when iter is low
                print("Shuffle data after an epoch! Should not happen when iter is low")
                rand_idx = torch.randperm(rays_rgb.shape[0])
                rays_rgb = rays_rgb[rand_idx]
                indices = indices[rand_idx]
                masks = masks[rand_idx]
                dynamic_rays_rgb = rays_rgb[torch.logical_not(masks[:, 0, 0])]
                static_rays_rgb = rays_rgb[masks[:, 0, 0]]
                i_dynamic = 0
                i_static = 0
                i_batch = 0

            dt = time.time() - time0
            global_step += 1

            if i % args.i_print == 0 and i > 0:
                psnr = mse2psnr(img2mse(rgb, target_s))
                tqdm.write(f"[TRAIN] Iter: {i} Loss: {loss.item():04f}  PSNR: {psnr.item():04f}")

        time_e0 = time.time()
        print(f"{time_e0-time_s0:.3f} sec / frame")

        # save trained frames
        img_i = i_val[0]
        gt_img = torch.Tensor(images[img_i]).to(device)
        pose = poses[img_i, :3, :4]

        with torch.no_grad():
            if args.pretraining_static:
                rgb, disp, acc, _ = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                                 **nerf[0]['render_kwargs_test'])

                psnr = mse2psnr(img2mse(rgb, gt_img))
                filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_{psnr.item():04f}.png')
                imageio.imwrite(filename, to8b(rgb.cpu().numpy()))
            else:
                rgb, disp, acc, extras = render_blend_two_models(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                                                 **render_kwargs_test_two_models)
                rgb_d, disp_d, acc_d, _ = render(H, W, K[img_i], chunk=args.chunk, c2w=pose,
                                                 **nerf[1]['render_kwargs_test'])

                psnr = mse2psnr(img2mse(rgb, gt_img))
                filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_{psnr.item():04f}.png')
                imageio.imwrite(filename, to8b(rgb.cpu().numpy()))
                filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_{psnr.item():04f}_dyn.png')
                imageio.imwrite(filename, to8b(rgb_d.cpu().numpy()))
                filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_{psnr.item():04f}_dyn_disp.png')
                imageio.imwrite(filename, to8b(disp_d.cpu().numpy()))
                filename = os.path.join(vis_dir, f'cam{img_i:02d}_frame{cur_frame:04d}_{psnr.item():04f}_dyn_acc.png')
                imageio.imwrite(filename, to8b(acc_d.cpu().numpy()))

        # Rest is logging
        suffix = ''
        for ni in range(len(nerf)):
            if args.pretraining_static and ni > 0:
                break

            suffix = '_static' if ni == 0 else '_dynamic'
            path = os.path.join(basedir, expname+suffix, f'{cur_frame:06d}{suffix}.tar')
            save_dict = {
                'global_step': global_step,
                'network_fn_state_dict': nerf[ni]['render_kwargs_train']['network_fn'].state_dict(),
                'network_fine_state_dict': nerf[ni]['render_kwargs_train']['network_fine'].state_dict(),
                'optimizer_state_dict': nerf[ni]['optimizer'].state_dict(),
            }

            torch.save(save_dict, path)
            print('Saved checkpoints at', path)

        # render video
        if cur_frame % args.i_img == 0 and args.render_videos:
            testsavedir = os.path.join(vis_dir, 'frame_{:04d}'.format(cur_frame))
            os.makedirs(testsavedir, exist_ok=True)
            with torch.no_grad():
                rgbs = torch.zeros(H, W, 3)  # [H,W,3] rendered image
                disps = torch.zeros(H, W)  # [H,W,3] rendered image
                for nr in nerf:
                    rgbs0, disps0 = render_path(render_poses[:60:2], hwf, K[6], args.chunk, nr['render_kwargs_test'], savedir=testsavedir)
                    rgb += rgbs0
                    disps += disps0
            print('Done, saving', rgbs.shape, disps.shape)


if __name__=='__main__':
    torch.set_default_tensor_type('torch.cuda.FloatTensor')

    train()
