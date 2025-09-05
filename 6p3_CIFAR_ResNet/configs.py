import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# FM_model
default_epoches = 600000
default_batchsize=300
default_sig_min = 0.000
default_lr = 1e-5 #2e-4: recommended lr for FM unet
FMmodel_name = 'FM_adv_unet'+'_iter'
FMsave_every = 10000
default_generation_step = 100


# RLFM model
RLFMsave_every = 20000
default_stage1_t = 0.8
default_RL_Steps_S = 15

default_batchsize_stage2 = 300
RLFMstage2_name = 'Apr24PGFM_0p815s'+'_iter'
plot_loss = True
plot_loss_every = 10
adv_reward = 10