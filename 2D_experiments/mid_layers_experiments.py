# Enable import from parent package
import sys
import os
sys.path.append( os.path.dirname( os.path.dirname( os.path.abspath(__file__) ) ) )

import torch, shutil, configargparse, glob, torchvision
import numpy as np
from tqdm import tqdm
import skvideo.io
from skvideo.io.ffmpeg import FFmpegReader
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from functools import partial
from PIL import Image

from demo import gon_model, ImageDataset, input_mapping, train_model
from siren_ff import SIRENFF
from helpers import vis_z_distribution


img_downsample = 4
n_layers = 5
mapping_size = 256
network_size = (n_layers, mapping_size*2, 256)
start_frame = 27
learning_rate = 1e-4
refine_iter = 400
is_vid = True
disable_mid_layers = False
use_intermediate_supervision = True
use_quant = False

# read video
if is_vid:
    # vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/near_still/couple_dancing.mp4'
    vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/family.mp4'
    reader = FFmpegReader(vid_path)
    n_frames, h, w, c = reader.getShape()
    data_iterator = reader.nextFrame()
else:
    # vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/kid_running/'
    vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/LF_snack/'
    frame_fn_list = sorted(glob.glob(os.path.join(vid_path, '*.jpg')))
    tmp_img = np.array(Image.open(frame_fn_list[0]))
    h, w = tmp_img.shape[:2]
    def frame_reader():
        for frame_fn in frame_fn_list:
            img = np.array(Image.open(frame_fn))
            yield img
    data_iterator = frame_reader()
image_resolution = (h//img_downsample, w//img_downsample)

model_random_mid = SIRENFF(*network_size).cuda()
model_ref = SIRENFF(*network_size).cuda()

model_dir = 'models_incremental'
model_fn_list = sorted(glob.glob(os.path.join(model_dir, '*.pth')))
B = torch.load(os.path.join(model_dir, '0000_B.pth'))

random_mid_layers_dir = 'with_n_without_training_mid_layers'
# if os.path.exists(random_mid_layers_dir):
#     assert False, "make sure you clean previous data"
# else:
#     os.makedirs(random_mid_layers_dir)

# setup model
# model_fn_cur = os.path.join(model_dir, f'{start_frame:04d}.pth')
# model_cur.load_state_dict(torch.load(model_fn_cur))
# model_ref.load_state_dict(torch.load(model_fn_cur))

# setup training
optim = torch.optim.Adam(model_random_mid.parameters(), lr=learning_rate)
loss_fn = torch.nn.MSELoss()

train_psnrs = []
for frame_i, frame in enumerate(data_iterator):
    if frame_i < (start_frame):
        continue

    # making data
    ds = ImageDataset(frame, (image_resolution[1], image_resolution[0]))
    grid, image = ds[0]
    grid = grid.unsqueeze(0).cuda()
    image = image.unsqueeze(0).cuda()
    test_data = (grid, image)
    train_data = (grid, image)

    if disable_mid_layers:
        for i, param in enumerate(model_random_mid.parameters()):
            if i < 2 or i >= 8:
                continue
            param.requires_grad = False
    out_updated = train_model(model_random_mid, optim, loss_fn, refine_iter, B,
                              train_data=train_data, test_data=(grid, image), device='cuda',
                              use_mid_supervision=use_intermediate_supervision)
    # model_cur = out_updated['model']
    img_updated = out_updated['img']
    loss = loss_fn(img_updated, image)
    cur_psnr = - 10 * torch.log10(2 * loss).item()
    torchvision.utils.save_image(img_updated.permute(0, 3, 1, 2),
                                 os.path.join(random_mid_layers_dir,
                                              f"{start_frame:04d}_{refine_iter}iter_{cur_psnr:.6f}_"
                                              f"{'freeze_mid_layers' if disable_mid_layers else 'train_mid_layers'}"
                                              f"{'_use_mid_supervision' if use_intermediate_supervision else ''}.jpeg"))

    # check only first layer changed
    # for l in range(0,3):
    #     print((model_ref.mid_layers[l].state_dict()['linear.weight'] - model_random_mid.mid_layers[l].state_dict()[
    #         'linear.weight']).abs().max())


    break





