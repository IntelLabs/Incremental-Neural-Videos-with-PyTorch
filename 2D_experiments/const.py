is_vid = True
if is_vid:
    # vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/near_still/couple_dancing.mp4'
    # vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/family.mp4'
    # vid_path = '/playpen-ssd/mikewang/incremental_neural_videos/2D_data/workout/workout.mp4'
    vid_path = '/playpen-ssd/mikewang/incremental_neural_videos/2D_data/family/'
    vid_name = 'family.mp4'
else:
    # vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/kid_running/'
    # vid_path = 'C:/Users/shengzew/OneDrive - Intel Corporation/Desktop/data/moving_camera/LF_snack/'
    vid_path = '/playpen-ssd/mikewang/incremental_neural_videos/2D_data/kid_running/'

is_nerf_baseline = False
per_frame_train_iters = 3000
if is_nerf_baseline:
    start_frame = 10   # frame indexing starts at 1
    train_iters = per_frame_train_iters * start_frame
else:
    start_frame = 1
    train_iters = per_frame_train_iters

n_layers = 5
img_downsample = 4
learning_rate = 1e-4

no_siren_only_mlp = True
use_nerf_pe = True
if use_nerf_pe:
    mapping_size = 128
    network_size = (5, (mapping_size*2)*2+2, 256)
else:
    mapping_size = 256
    network_size = (5, mapping_size*2, 256)




