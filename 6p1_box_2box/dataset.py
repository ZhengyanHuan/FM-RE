#%%
import numpy as np

import configs
from configs import *

# set_seed(seed)


dim = 2
dataset_type = 'uniform2'# uniform, cropped_Gaussian
dataset_size = int(5e6)

def gen_uniform(bounds, sample_num):
    out = np.array([])
    for i in range(len(bounds)):
        this_dim_sample  = np.random.uniform(bounds[i,0],  bounds[i,1], sample_num)

        if i == 0:
            out = this_dim_sample
        else:
            out = np.vstack([out, this_dim_sample])
    return out.T



def gen_gaussian(mean, var, sample_num):
    out = np.array([])
    for i in range(len(mean)):
        this_dim_sample  = np.random.normal(loc = mean[i],  scale = np.sqrt(var[i]), size = sample_num)

        if i == 0:
            out = this_dim_sample
        else:
            out = np.vstack([out, this_dim_sample])
    return out.T

def crop(samples, bounds):
    for i in range(len(bounds)):
        if i == 0:
            flag = (samples[:,i] <= bounds[i,1]) * (samples[:,i] >= bounds[i,0])
        else:
            flag = flag * (samples[:,i] <= bounds[i,1]) * (samples[:,i] >= bounds[i,0])

    return samples[flag]


if dim == 2:
    if dataset_type == 'uniform':
        bounds = np.zeros([2,2])
        bounds[0, 0] = - configs.bound + configs.uniform_center
        bounds[0, 1] = configs.bound + configs.uniform_center
        bounds[1, 0] = - configs.bound + configs.uniform_center
        bounds[1, 1] = configs.bound + configs.uniform_center

        xy = gen_uniform(bounds, dataset_size)
    elif dataset_type == 'uniform2':
        bounds = np.zeros([2,2])
        bounds[0, 0] = - configs.bound + configs.uniform_center
        bounds[0, 1] = configs.bound + configs.uniform_center
        bounds[1, 0] = - configs.bound + configs.uniform_center
        bounds[1, 1] = configs.bound + configs.uniform_center

        xy1 = gen_uniform(bounds, dataset_size)
        
        bounds = np.zeros([2,2])
        bounds[0, 0] = - configs.bound - configs.uniform_center
        bounds[0, 1] = configs.bound - configs.uniform_center
        bounds[1, 0] = - configs.bound - configs.uniform_center
        bounds[1, 1] = configs.bound - configs.uniform_center

        xy2 = gen_uniform(bounds, dataset_size)
        xy = np.vstack([xy1, xy2])
        np.random.shuffle(xy)
        
        
    elif dataset_type == 'cropped_Gaussian':
        G_center = np.array([[configs.bound*0.75+ configs.uniform_center, configs.bound*0.75+ configs.uniform_center,],[-configs.bound*0.75+ configs.uniform_center,-configs.bound*0.75+ configs.uniform_center]])
        # G_center = np.array(
        #     [[configs.bound  + configs.uniform_center, configs.bound + configs.uniform_center, ],
        #      [-configs.bound  + configs.uniform_center, -configs.bound  + configs.uniform_center]])

        # G_center = np.array(
        #     [[0, 0 ],
        #      [0, 0]])

        G_var = np.array([[0.6,0.6],[1.5,1.5]])
        G_p = np.array([0.5,0.5])

        bounds = np.zeros([2,2])
        bounds[0, 0] = - configs.bound + configs.uniform_center
        bounds[0, 1] = configs.bound + configs.uniform_center
        bounds[1, 0] = - configs.bound + configs.uniform_center
        bounds[1, 1] = configs.bound + configs.uniform_center

        for i in range(len(G_p)):
            xy_tmp = np.array([])
            dataset_size_part = int(dataset_size * G_p[i])
            remain_size = dataset_size_part
            while remain_size > 0:
                G_part = gen_gaussian(G_center[i], G_var[i], dataset_size_part)
                G_part_crop = crop(G_part, bounds)
                # G_part_crop = G_part

                if xy_tmp.shape[0] == 0:
                    xy_tmp = G_part_crop[: min(remain_size, len(G_part_crop)),:]
                else:
                    xy_tmp = np.vstack([xy_tmp, G_part_crop[: min(remain_size, len(G_part_crop)),:]])
                remain_size = remain_size - G_part_crop.shape[0]
            if i == 0:
                xy = xy_tmp
            else:
                xy = np.vstack([xy, xy_tmp])
        np.random.shuffle(xy)


np.save(r'./data/'+dataset_type + ".npy", xy)