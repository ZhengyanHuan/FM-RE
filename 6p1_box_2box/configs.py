import torch
import numpy as np


# general
seed = 12
output_dim = 2

#dataset
dataset = "cropped_Gaussian" # two_uniform_box cropped_Gaussian uniform_box

if dataset == "cropped_Gaussian":
    bound = 4  # 2 4
    uniform_center = 0  # 3 0
else:
    bound = 2#2 4
    uniform_center = 3#3 0

# FM_model
device = 'cuda:0'
default_epoches = 5000
default_batchsize=10000
default_sig_min = 0.001
default_lr = 6e-5
FMmodel_name = 'FM_2uniform'
save_every = 5000
default_generation_step = 50


# RLFM model
default_stage1_t = 0.8
default_RL_Steps_S = 15
default_constraint_reward = 200#80
default_batchsize_stage2 = 10000
RLFMstage2_name = 'distanceFM_2uniform'#'Feb27RLFM_CG_s2'
plot_loss = True
plot_loss_every = 10

# # baseline
# baseline_lr = 5e-5
# baseline_epoches = 3000
# baseline_save_every = 3000
# baseline_name = 'baseline_CG10'

def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)


def min_distance2box(x_N2, func_center, func_bound):
    distance = torch.zeros((x_N2.shape[0], 8)).to(device)
    corners = torch.tensor([[func_center+func_bound, func_center + func_bound],
                        [func_center+func_bound, func_center-func_bound],
                        [func_center-func_bound, func_center+func_bound],
                        [func_center-func_bound, func_center-func_bound]]).to(device)
    distance[:,:4] = torch.norm(x_N2[:, None, :] - corners[None, :, :], dim=2)
    bounds = torch.tensor([func_center+func_bound,
                       func_center-func_bound,
                       func_center+func_bound,
                       func_center-func_bound]).to(device)

    distance[:, 4] = torch.abs(x_N2[:, 0] - bounds[0])  # Distance to x = c[0]
    distance[:, 5] = torch.abs(x_N2[:, 0] - bounds[1])  # Distance to x = c[1]
    distance[:, 6] = torch.abs(x_N2[:, 1] - bounds[2])  # Distance to y = c[2]
    distance[:, 7] = torch.abs(x_N2[:, 1] - bounds[3])  # Distance to y = c[3]
    out = torch.min(distance, dim = 1)[0]*default_constraint_reward
    return out

if dataset == "two_uniform_box":
    def distance(x_N2):
        distance1 = min_distance2box(x_N2, uniform_center, bound)
        distance2 = min_distance2box(x_N2, -uniform_center, bound)
        return torch.minimum(distance1, distance2)
else:
    def distance(x_N2):
        return min_distance2box(x_N2, uniform_center, bound)