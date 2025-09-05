import torch
import numpy as np


# general
seed = 12
d_model = 8 # 8 or 20



# FM_model
device = 'cuda:0'
default_epoches = 200000
default_batchsize=10000
default_sig_min = 0.000
default_lr = 1e-4#5e-6
FMmodel_name = 'RFM_l2ball_dim'+str(d_model)+'_iter'
FMsave_every = 100000
default_generation_step = 100


# RLFM model
RLFMsave_every = 200000
default_stage1_t = 0.8
default_RL_Steps_S = 15
default_constraint_reward = 20
default_batchsize_stage2 = 10000
RLFMstage2_name = 'DDFM_l2balls2_'+str(default_constraint_reward)+'_dim'+str(d_model)+'_iter'
plot_loss = True
plot_loss_every = 10



def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)