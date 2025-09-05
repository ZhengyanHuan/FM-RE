
import torch
import numpy as np

d_model = 10

# FM_model
device = 'cuda:0'
default_epoches = 20000
default_batchsize=10000
default_sig_min = 0.001
default_lr = 1e-6
FMmodel_name = 'FM_subspace'
FM_save_every = 500000
default_generation_step = 100

# RLFM
RLFMsave_every = 10000
default_stage1_t = 0.9
default_RL_Steps_S = 10
default_constraint_reward = 1
default_batchsize_stage2 = 10000
RLFMstage2_name = 'FM2_subspace'
plot_loss = True
plot_loss_every = 100
