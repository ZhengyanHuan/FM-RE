#%%
import abc
import numpy as np

import torch
import torch.distributions as td
import torch.nn.functional as F

# from . import constraintset
# from .constraintset import BALL_RADIUS
#
# from ipdb import set_trace as debug

data_type = "l2ball_GMM"
dataset_size = int(1e6)
xdim = 20
#%%



def check_inside_l2ball(mat_Ndim):
    return torch.norm(mat_Ndim, dim=-1)<=1

#%%

if __name__ == "__main__":
    if data_type == "l2ball_GMM":
        sigma = 0.05

        means = F.one_hot(torch.arange(xdim), num_classes=xdim).float()
        var = sigma * torch.ones(xdim, xdim)

        mix = td.Categorical(torch.ones(xdim, ))
        comp = td.Independent(td.Normal(
            torch.Tensor(means),
            var),
            1
        )

        distribution = td.MixtureSameFamily(mix, comp)

        out = torch.tensor([])

        while out.shape[0] < dataset_size:
            data_sec = distribution.sample([dataset_size])
            inside_mask = check_inside_l2ball(data_sec)
            print(torch.sum(inside_mask)/dataset_size)
            if out.shape[0] == 0:
                out = data_sec[inside_mask]
            else:
                out = torch.cat([out, data_sec[inside_mask]], dim =0)

        xy = out[:dataset_size]
        xy_np =xy.numpy()

        np.save(r'./data/'+ 'MDMl2ball_dim'+str(xdim) + ".npy", xy)