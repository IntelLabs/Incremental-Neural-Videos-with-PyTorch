import torch, torchvision, glob, os
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from PIL import Image
from skvideo.io.ffmpeg import FFmpegReader

from dataset import ImageDataset

import matplotlib.pyplot as plt


class Swish(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x * torch.sigmoid(x)


class SirenLayer(nn.Module):
    def __init__(self, in_f, out_f, w0=30, is_first=False, is_last=False):
        super().__init__()
        self.in_f = in_f
        self.w0 = w0
        self.linear = nn.Linear(in_f, out_f)
        self.is_first = is_first
        self.is_last = is_last
        self.init_weights()

    def init_weights(self):
        b = 1 / \
            self.in_f if self.is_first else np.sqrt(6 / self.in_f) / self.w0
        with torch.no_grad():
            self.linear.weight.uniform_(-b, b)

    def forward(self, x):
        x = self.linear(x)
        return x if self.is_last else torch.sin(self.w0 * x)


def input_mapping(x, B):
    if B is None:
        return x
    else:
        x_proj = (2. * np.pi * x) @ B.t()
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


def make_network(num_layers, input_dim, hidden_dim):
    layers = [nn.Linear(input_dim, hidden_dim), Swish()]
    for i in range(1, num_layers - 1):
        layers.append(nn.Linear(hidden_dim, hidden_dim))
        layers.append(Swish())

    layers.append(nn.Linear(hidden_dim, 3))
    layers.append(nn.Sigmoid())
    return nn.Sequential(*layers)


def gon_model(num_layers, input_dim, hidden_dim):
    layers = [SirenLayer(input_dim, hidden_dim, is_first=True)]
    for i in range(1, num_layers - 1):
        layers.append(SirenLayer(hidden_dim, hidden_dim))
    layers.append(SirenLayer(hidden_dim, 3, is_last=True))

    return nn.Sequential(*layers)


def train_model(model, optim, loss_fn, iters, B, train_data, test_data, use_mid_supervision=False,
                device="cpu"):
    train_psnrs = []
    # test_psnrs = []
    # xs = []
    for i in tqdm(range(iters), desc='train iter', leave=False):
        model.train()
        optim.zero_grad()

        t_o, mid_o = model(input_mapping(train_data[0], B))
        t_loss = .5 * loss_fn(t_o, train_data[1])
        # if use_mid_supervision:
        #     t_loss += .5 * loss_fn(mid_o, train_data[1])

        # t_loss.retain_grad()
        # t_o.retain_grad()
        # model.final_layer.linear.bias.retain_grad()
        t_loss.backward()
        optim.step()

        # print(f"---[steps: {i}]: train loss: {t_loss.item():.6f}")

        train_psnrs.append(- 10 * torch.log10(2 * t_loss).item())

        # if i % 100 == 0:
    model.eval()
    with torch.no_grad():
        v_o, mid_o = model(input_mapping(test_data[0], B))
        v_loss = loss_fn(v_o, test_data[1])
        v_psnrs = - 10 * torch.log10(2 * v_loss).item()
        # test_psnr = v_psnrs
        # xs.append(i)
        # torchvision.utils.save_image(v_o.permute(0, 3, 1, 2), f"imgs/{i}_{v_loss.item():.6f}.jpeg")
        # torchvision.utils.save_image(v_o.permute(0, 3, 1, 2), f"imgs/{i}_{v_psnrs:.6f}.jpeg")
    # print(f"---[steps: {i}]: valid loss: {v_loss.item():.6f}")

    return {
        'state': model.state_dict(),
        'train_psnrs': train_psnrs,
        'test_psnrs': v_psnrs,
        'img': v_o,
        'model': model
    }


if __name__ == '__main__':
    device = "cuda:0"

    img_downsample = 4
    network_size = (5, 512, 256)
    learning_rate = 1e-4
    iters = 200
    mapping_size = 256
    is_vid = True

    B_gauss = torch.randn((mapping_size, 2)).to(device) * 10

    # ds = ImageDataset("data/fox.jpg", 512)
    result_dir = 'imgs_incremental'
    gt_dir = 'imgs_incremental_gt'
    model_dir = 'models_incremental'
    if os.path.exists(model_dir) or os.path.exists(gt_dir) or os.path.exists(result_dir):
        assert False, "make sure you clean previous data"
    else:
        os.makedirs(result_dir)
        os.makedirs(gt_dir)
        os.makedirs(model_dir)

    # read video
    if is_vid:
        vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/family.mp4'
        reader = FFmpegReader(vid_path)
        n_frames, h, w, c = reader.getShape()
        data_iterator = reader.nextFrame()
    else:
        vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/kid_running/'
        frame_fn_list = sorted(glob.glob(os.path.join(vid_path, '*.png')))
        tmp_img = np.array(Image.open(frame_fn_list[0]))
        h, w = tmp_img.shape[:2]
        def frame_reader():
            for frame_fn in frame_fn_list:
                img = np.array(Image.open(frame_fn))
                yield img
        data_iterator = frame_reader()
    image_resolution = (h // img_downsample, w // img_downsample)

    # setup optimizer and model
    model = gon_model(*network_size).to(device)
    optim = torch.optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = torch.nn.MSELoss()

    for idx, frame in enumerate(data_iterator):

        # making data
        ds = ImageDataset(frame, (image_resolution[1], image_resolution[0]))
        grid, image = ds[0]
        grid = grid.unsqueeze(0).to(device)
        image = image.unsqueeze(0).to(device)
        test_data = (grid, image)
        # train_data = (grid[:, ::2, ::2], image[:, ::2, :: 2])
        train_data = (grid, image)

        output = train_model(model, optim, loss_fn, iters, B_gauss,
                             train_data=train_data, test_data=(grid, image), device=device)
        img_out = output['img'].permute(0,3,1,2)
        test_psnr = output['test_psnrs']
        weights = output['state']
        torchvision.utils.save_image(img_out, os.path.join(result_dir, f"{idx:04d}_{test_psnr:.6f}.jpeg"))
        torch.save(weights, os.path.join(model_dir, f'{idx:04d}.pth'))
        torch.save(B_gauss, os.path.join(model_dir, f'{idx:04d}_B.pth'))
