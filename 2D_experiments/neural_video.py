import torch, torchvision, glob, os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from skvideo.io.ffmpeg import FFmpegReader

from const import *
from dataset import ImageDataset
from siren_ff import SIRENFF, input_mapping, MLP, get_embedder

import matplotlib.pyplot as plt


def train_model(model, optim, loss_fn, iters, B, use_nerf_pe, train_data, test_data,
                device="cpu", frame=None):
    train_psnrs = []
    for i in tqdm(range(iters), desc='train iter', leave=False):
        model.train()
        optim.zero_grad()

        t_o,_ = model(input_mapping(train_data[0], B, use_nerf_pe))
        t_loss = .5 * loss_fn(t_o, train_data[1])

        t_loss.backward()
        optim.step()

        train_psnrs.append(- 10 * torch.log10(2 * t_loss).item())

        if is_baseline and (i+1)%per_frame_train_iters == 0:
            model.eval()
            with torch.no_grad():
                v_o, _ = model(input_mapping(test_data[0], B, use_nerf_pe))
                v_loss = loss_fn(v_o, test_data[1])
                v_psnrs = - 10 * torch.log10(2 * v_loss).item()
                torchvision.utils.save_image(v_o.permute(0,3,1,2), os.path.join(result_dir, f"{frame:04d}_{i:06d}_{v_psnrs:.6f}.jpeg"))


    model.eval()
    with torch.no_grad():
        v_o, _ = model(input_mapping(test_data[0], B, use_nerf_pe))
        v_loss = loss_fn(v_o, test_data[1])
        v_psnrs = - 10 * torch.log10(2 * v_loss).item()

    return {
        'state': model.state_dict(),
        'train_psnrs': train_psnrs,
        'test_psnrs': v_psnrs,
        'img': v_o,
        'model': model
    }


if __name__ == '__main__':
    device = "cuda:0"

    if is_baseline:
        # for quantitative comparisons between
        # [incremental frame-to-frame] and [baseline (from scratch)] at the same accumulative iterations
        result_dir = os.path.join(vid_path, 'imgs_baseline')
        model_dir = os.path.join(vid_path, 'models_baseline')
        if os.path.exists(model_dir) or os.path.exists(result_dir):
            assert False, "make sure you clean previous data"
        else:
            os.makedirs(result_dir)
            os.makedirs(model_dir)
    else:
        result_dir = os.path.join(vid_path, 'imgs_incremental')
        gt_dir = os.path.join(vid_path, 'imgs_incremental_gt')
        model_dir = os.path.join(vid_path, 'models_incremental')
        if os.path.exists(model_dir) or os.path.exists(gt_dir) or os.path.exists(result_dir):
            assert False, "make sure you clean previous data"
        else:
            os.makedirs(result_dir)
            os.makedirs(gt_dir)
            os.makedirs(model_dir)

    # read video
    if is_vid:
        # read frames from video file
        reader = FFmpegReader(os.path.join(vid_path, vid_name))
        n_frames, h, w, c = reader.getShape()
        data_iterator = reader.nextFrame()
    else:
        # read images extracted from the videos
        frame_fn_list = sorted(glob.glob(os.path.join(vid_path, 'frames', '*.png')))
        tmp_img = np.array(Image.open(frame_fn_list[0]))
        h, w = tmp_img.shape[:2]
        def frame_reader():
            for frame_fn in frame_fn_list:
                img = np.array(Image.open(frame_fn))
                yield img
        data_iterator = frame_reader()
    image_resolution = (h // img_downsample, w // img_downsample)

    # select the model and positional encoding
    # Note: 4 types of 2D mlp are supported
    #       (1) simple mlp + nerf pos. enc. (sin & cos encodes xyz separately)
    #       (2) SIREN (sin activation) + nerf pos. enc.
    #       (3) simple mlp + FF (fourier features, sin & cos encodes xyz together)
    #       (4) SIREN + FF
    #       SIREN + FF usually gives best results, and best for structure swap.

    # Note: without positional encoding, we don't notice structure layers since
    #       structure swap fails. Color/structure info is mixed in all layers,
    #       instead of structure info being store mostly in the 1st layer.
    #       Thus, we speculate that the improvement from positional encoding could be tied to
    #       the good separation between structure and color information
    if use_nerf_pe:
        # Pos. Enc. used by NeRF. Sin & cos encoding for x, y, and z separately
        # Note: In both 2D & 3D, we notice stripe like artifacts during training,
        #  possibly due to the separate encoding
        B_emb, _ = get_embedder(mapping_size)
    else:
        # Fourier Features. Sin & cos encodes xyz together using matrix multiplication
        # Note: In both 2D & 3D, we notice blob like artifacts during training,
        #  possibly due to encoding xyz's together
        B_emb = torch.randn((mapping_size, 2)).to(device) * 10

    if no_siren_only_mlp:
        model = MLP(*network_size).to(device)
    else:
        # SIREN uses sin activation layers
        model = SIRENFF(*network_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for idx, frame in enumerate(data_iterator):
        if idx<start_frame:
            continue

        # making data
        ds = ImageDataset(frame, (image_resolution[1], image_resolution[0]))
        grid, image = ds[0]
        grid = grid.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)
        test_data = (grid, image)
        train_data = (grid, image)

        output = train_model(model, optim, loss_fn, train_iters, B_emb, use_nerf_pe,
                             train_data=train_data, test_data=(grid, image), device=device, frame=idx)
        img_out = output['img'].permute(0,3,1,2)
        test_psnr = output['test_psnrs']
        weights = output['state']
        torchvision.utils.save_image(img_out, os.path.join(result_dir, f"{idx:04d}_{test_psnr:.6f}.jpeg"))
        torch.save(weights, os.path.join(model_dir, f'{idx:04d}.pth'))
        if not use_nerf_pe:
            torch.save(B_emb, os.path.join(model_dir, f'{idx:04d}_B.pth'))
