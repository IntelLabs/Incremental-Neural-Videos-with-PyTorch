is_vid = True
if is_vid:
    vid_path = '/playpen-ssd/mikewang/incremental_neural_videos/2D_data/family/'
    vid_name = 'family.mp4'
else:
    vid_path = '/playpen-ssd/mikewang/incremental_neural_videos/2D_data/kid_running/'

is_baseline = False
per_frame_train_iters = 3000
if is_baseline:     # for quantitative comparisons at the same iterations
    start_frame = 10   # frame indices starts at 1
    train_iters = per_frame_train_iters * start_frame
else:
    start_frame = 1
    train_iters = per_frame_train_iters

n_layers = 5
img_downsample = 4
learning_rate = 1e-4

no_siren_only_mlp = False
use_nerf_pe = False
if use_nerf_pe:
    mapping_size = 128
    network_size = (5, (mapping_size*2)*2+2, 256)
else:
    mapping_size = 256
    network_size = (5, mapping_size*2, 256)




