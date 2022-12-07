import torch, torchvision, glob, os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from skvideo.io.ffmpeg import FFmpegReader

from const import *
from dataset import ImageDataset
from siren_ff import SIRENFF, input_mapping, FFN, get_embedder

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

        if is_nerf_baseline and (i+1)%per_frame_train_iters == 0:
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

    # parameters
    if use_nerf_pe:
        B_emb, _ = get_embedder(mapping_size)
    else:
        B_emb = torch.randn((mapping_size, 2)).to(device) * 10

    # ds = ImageDataset("data/fox.jpg", 512)
    if is_nerf_baseline:
        result_dir = os.path.join(vid_path, 'imgs_nerf_baseline')
        model_dir = os.path.join(vid_path, 'models_nerf_baseline')
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

    # setup optimizer and model
    if no_siren_only_mlp:
        model = FFN(*network_size).to(device)
    else:
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
