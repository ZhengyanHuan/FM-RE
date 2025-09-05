import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# FM_model
default_epoches = 1000000
default_batchsize=1000
default_sig_min = 0#0.001
default_lr = 1e-5
FMmodel_name = 'FM_adv'+'_iter'
FMsave_every = 50000
default_generation_step = 100


# RLFM model
RLFMsave_every = 10000
default_stage1_t = 0.8
default_RL_Steps_S = 15

default_batchsize_stage2 = 300
RLFMstage2_name = 'Apr23RLFM_advs207'+'_iter'
plot_loss = True
plot_loss_every = 10
adv_reward = 1