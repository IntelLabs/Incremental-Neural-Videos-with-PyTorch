import glob
import numpy as np
import cv2
import os

# import lpips
# from PerceptualSimilarity
import torch

# loss_fn = lpips.LPIPS(net='alex-lin').cuda()

import models
def im2tensor(image, imtype=np.uint8, cent=1., factor=1./2.):
    return torch.Tensor((image / factor - cent)
                        [:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
model = models.PerceptualLoss(model='net-lin',net='alex',use_gpu=True,version=0.1)

import matplotlib.pyplot as plt

n_imgs = 11
fps = 30
DO_NERF = True
partial_xfer_start = 150
DO_NGP = False
DO_MERGE_GRAPH = False
DO_PSNR_LPIPS_ONE_PLOT = True
DO_MAKE_VIDEO = False

# root_dir = '/playpen-ssd/mikewang/incremental_neural_videos/META_data/flame_salmon_1/down_2x/'
# base_folder_list = [f'{root_dir}' + 'META_flame_salmon_1_warmup10k_iter10k_s3_stability0_relu_l2_freeze120/']
# base_folder_list.append(f'{root_dir}' + 'META_flame_salmon_1_warmup20k_iter20k_s3_stability0_relu_l1_no_skip/')

# base_folder_list = [f'{root_dir}' + 'META_flame_salmon_1_warmup120k_iter120k_s3_stability0_SD65_relu_no_skip/']
# base_folder_list.append(f'{root_dir}' + 'META_flame_salmon_1_warmup120k_iter120k_s3_stability0_SD65_relu_no_skip/')
# results_folder = ['nerf_esti', 'compressed_struct_weights_270']

root_dir = '/playpen-ssd/mikewang/incremental_neural_videos/META_data/coffee_martini/down_2x/'
base_folder_list = [f'{root_dir}' + 'META_coffee_martini_100k_baselineNeRF_frame10/']
base_folder_list.append(f'{root_dir}' + 'META_coffee_martini_warmup10k_iter10k_s3/')
results_folder = ['nerf_esti', 'nerf_esti']


out = {}
for i, base_folder in enumerate(base_folder_list):
    expname = base_folder.split('/')[-2].split('_')[5].replace('iter', '') + ' iter/frame'
    if i == (len(base_folder_list)-1):
        expname = base_folder.split('/')[-2].split('_')[5].replace('iter', '') + ' iter/frame w/ trained color layers'
    ngp_frames_folder = base_folder+'frames'
    ngp_image_folder = base_folder+'instant_ngp_esti'
    inv_frames_folder = base_folder+'../frames_2'
    # inv_image_folder = base_folder+'nerf_esti'
    inv_image_folder = base_folder+results_folder[i]

    ngp_images = sorted(glob.glob(os.path.join(ngp_image_folder, "*.jpg")))
    ngp_frames = sorted(glob.glob(os.path.join(ngp_frames_folder, "*cam00.png")))
    inv_images = sorted(glob.glob(os.path.join(inv_image_folder, "*.png")))
    # gt_frames = sorted(glob.glob(os.path.join(inv_frames_folder, "*cam00.png")))

    height, width, layers = cv2.imread(inv_images[0]).shape
    inv_psnr_list = []
    inv_LPIPS_list = []
    out[base_folder_list[i]] = {}
    for fi, image in enumerate(inv_images):
        if fi >= n_imgs:
            break
        print(f"{fi}/{len(inv_images)}")
        esti = cv2.imread(image)
        gt = cv2.imread(os.path.join(inv_frames_folder, "frame0010cam00.png"))
        esti = im2tensor(esti).cuda() / 255
        gt = im2tensor(gt).cuda() / 255

        cur_psnr = float(image.split('_')[-1][:-4])
        with torch.no_grad():
            lpips_loss = model.forward(gt, esti).item()
        print(f"cur_psnr: {cur_psnr:.6f}, LPIPS: {lpips_loss:.6f}")
        inv_psnr_list.append(cur_psnr)
        inv_LPIPS_list.append(lpips_loss)

    print(f"mean psnr: {np.mean(inv_psnr_list):.6f}, LPIPS: {np.mean(inv_LPIPS_list):.6f}")
    out[base_folder_list[i]]['nerf_psnr_list'] = inv_psnr_list
    out[base_folder_list[i]]['nerf_LPIPS_list'] = inv_LPIPS_list
    out[base_folder_list[i]]['expname'] = expname

if DO_NGP:
    ngp_psnr_list = []
    for i, (image, frame) in enumerate(zip(ngp_images, ngp_frames)):
        print(f"{i}/{len(ngp_images)}")
        a = cv2.imread(image)
        b = cv2.imread(frame)
        cur_psnr = cv2.PSNR(a, b)
        ngp_psnr_list.append(cur_psnr)
    out['ngp_psnr_list'] = ngp_psnr_list

if DO_PSNR_LPIPS_ONE_PLOT:
    fig, ax1 = plt.subplots()
    ax1.set_xlabel('frame number', fontsize=14, fontweight="bold")
    ax1.set_ylabel('PSNR', fontsize=14, fontweight="bold")
    def forward(x):
        is_zero = (x==0)
        out = x ** (1 / 2)
        out[is_zero] = 0
        return out
    def inverse(x):
        return x ** 2
    ax1.set_xscale('function', functions=(forward, inverse))
    def forward_y(x):
        return x ** 1.6
    def inverse_y(x):
        return x ** (1 / 1.6)
    ax1.set_yscale('function', functions=(forward_y, inverse_y))
    ax1.plot(list(range(0,100001,10000)), out[base_folder_list[0]]['nerf_psnr_list'], 'g--',
             label='NeRF 100k from scratch (PSNR)')
    ax1.plot(list(range(0,10001,1000)), out[base_folder_list[1]]['nerf_psnr_list'], 'g',
             label='I.T. 10k from frame 9 (PSNR)')
    # plt.legend(fontsize=14)
    # Adding Twin Axes to plot using dataset_2
    ax2 = ax1.twinx()

    color = 'tab:green'
    ax2.set_ylabel('LPIPS', color=color, fontsize=14, fontweight="bold")
    ax2.set_yscale('log')
    ax2.plot(list(range(0, 100001, 10000)), out[base_folder_list[0]]['nerf_LPIPS_list'], 'b--',
             label='NeRF 100k from scratch (LPIPS)', lw=2)
    ax2.plot(list(range(0, 10001, 1000)), out[base_folder_list[1]]['nerf_LPIPS_list'], 'b',
             label='I.T. 10k from frame 9 (LPIPS)', lw=2)
    ax2.tick_params(axis='y', labelcolor=color)

    # Adding title
    plt.title(f'INV (from frame 9) Vs. NeRF (from scratch)',
              fontweight="bold", fontsize=16)
    # plt.legend(fontsize=14)
    # xticks = ax1.xaxis.get_major_ticks()
    # xticks[0].label.set_visible(True)
    plt.xlim([0, 100000])
    plt.subplots_adjust(right=0.85)
    plt.show()

if DO_NGP and DO_MERGE_GRAPH:
    title_str = f'INV mean: {np.mean(inv_psnr_list):.3f}, median{np.median(inv_psnr_list):.3f}\n'+\
                f'NGP mean: {np.mean(ngp_psnr_list):.3f}, median{np.median(ngp_psnr_list):.3f}\n'
    plt.plot(list(range(len(inv_psnr_list))), inv_psnr_list, color='g', label='INV')
    plt.plot(list(range(len(ngp_psnr_list))), ngp_psnr_list, color='r', label='NGP + Incre Xfer')
    plt.xlabel('frame #')
    plt.ylabel('PSNR')
    plt.ylim([15, 35])
    plt.title(title_str)
    plt.legend()
    # function to show the plot
    plt.show()

np.save(base_folder+'all_qualitative.npy', out)
