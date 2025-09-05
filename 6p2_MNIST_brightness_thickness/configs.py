import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'


#0 2.5
min_thickness = 2
max_thickness = 3

min_brightness = 100
max_brightness = 28*28
# FM_model
default_epoches = 50000
default_batchsize=1000
default_sig_min = 0.001
default_lr = 3e-5
FMmodel_name = 'FM_MNIST_bright'+'_iter'
FMsave_every = 40000
default_generation_step = 60


# RLFM model
RLFMsave_every = 10000
default_stage1_t = 0.6
default_RL_Steps_S = 20

default_batchsize_stage2 = 500
RLFMstage2_name = 'Apr25RLFM_MNIST_bright_06s20'+'_iter'
plot_loss = True
plot_loss_every = 10
constrained_reward = 2


def count_valid_num_thick(thickness_vec):
    valid_ind = np.where((thickness_vec > min_thickness)&(thickness_vec<max_thickness))[0]
    return valid_ind, len(valid_ind)


def count_valid_num_bright(brightness_vec):
    valid_ind = np.where((brightness_vec > min_brightness)&(brightness_vec<max_brightness))[0]
    return valid_ind, len(valid_ind)