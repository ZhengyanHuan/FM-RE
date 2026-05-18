import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import configs
# from FMfuncs import FMmodel
# from Unet import UNet

class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class RLs2Model(nn.Module):
    # https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm

    def __init__(self, channels=[64, 128, 256, 512], embed_dim=256):
        """Initialize a time-dependent score-based network.

        Args:
          marginal_prob_std: A function that takes time t and gives the standard
            deviation of the perturbation kernel p_{0t}(x(t) | x(0)).
          channels: The number of channels for feature maps of each resolution.
          embed_dim: The dimensionality of Gaussian random feature embeddings.
        """
        super().__init__()
        # Gaussian random feature embedding layer for time
        self.embed = nn.Sequential(nn.Linear(1, embed_dim),
                                   nn.ReLU(),
                                   nn.Linear(embed_dim, embed_dim))
        # Encoding layers where the resolution decreases
        self.conv1 = nn.Conv2d(3, channels[0], 3, stride=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gnorm1 = nn.GroupNorm(4, num_channels=channels[0])
        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gnorm2 = nn.GroupNorm(32, num_channels=channels[1])
        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gnorm3 = nn.GroupNorm(32, num_channels=channels[2])
        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gnorm4 = nn.GroupNorm(32, num_channels=channels[3])

        # Decoding layers where the resolution increases
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, bias=False, output_padding=1)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgnorm4 = nn.GroupNorm(32, num_channels=channels[2])
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        ####path1: mean
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False,
                                         output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 3, 3, stride=1)
        ####path2: var
        # self.tconv2v = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        # self.dense7v = Dense(embed_dim, channels[0])
        # self.tgnorm2v = nn.GroupNorm(32, num_channels=channels[0])
        # self.tconv1v = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        # self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t):
        # Obtain the Gaussian random feature embedding for t
        # x = torch.cat((x, x1), dim=1)
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)

        ####path2: var
        #         v = self.tconv2v(torch.cat([h, h2], dim=1))
        #         v += self.dense7v(embed)
        #         v = self.tgnorm2v(v)
        #         v = self.act(v)
        #         v = self.tconv1v(torch.cat([v, h1], dim=1))
        #         v = 1e-5+torch.exp(v)
        #         v[v>1e-1] = 1e-1

        ####path1: mean
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))


        return h

class PolicyNet(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self):
        super(PolicyNet, self).__init__()
        self.net = RLs2Model()
        self.log_std = nn.Parameter(torch.zeros(32*32))

    def forward(self, x, t):
        mean = self.net(x, t)
        std = torch.exp(self.log_std)+1e-5 #control the initial exploration range
        return mean, std.view(1, 32, 32)

    def get_v(self, cur_state_N1DD, cur_t_N, deterministic=False):
        # print(cur_t_N[:, None].shape)
        mean, std = self.forward(cur_state_N1DD, cur_t_N[:, None])
        if deterministic:
            v_ND_grad = mean
            v_ND = v_ND_grad.detach()
            log_prob_N = torch.zeros_like(mean).sum(-1)
        else:
            dist = torch.distributions.Normal(mean, std)
            v_ND_grad = mean + std * torch.randn_like(mean)
            v_ND = v_ND_grad.detach()
            log_prob_N = dist.log_prob(v_ND).sum(dim=[1,2,3])
        return v_ND_grad, v_ND, log_prob_N



class PGFM:

    def __init__(self, bb_model, sig_min: float = configs.default_sig_min, stage1_t=configs.default_stage1_t,
                 RL_Steps_S=configs.default_RL_Steps_S
                 , device=configs.device, adv_reward = configs.adv_reward) :
        super().__init__()
        self.sig_min = sig_min
        self.bb_model = bb_model
        self.crieria = nn.MSELoss()
        self.stage1_t = stage1_t
        self.RL_Steps_S = RL_Steps_S
        self.device = device
        self.RL_step_width = (1 - self.stage1_t) / (self.RL_Steps_S)
        self.policy = PolicyNet().to(self.device)
        # self.policy_wref = PolicyNet_wref().to(self.device)
        self.adv_reward = adv_reward

    def get_untrained_model(self):
        # torch.manual_seed(42)
        # REF https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/models.py
        return RLs2Model().to(configs.device)


    def sample_xt_given_x1_x0(self, x0_ND: torch.Tensor, x1_ND: torch.Tensor, t_N: torch.Tensor, sig_min = None):
        if sig_min is None:
            std1 = self.sig_min
        else:
            std1 = sig_min
        return (1 - (1 - std1) * t_N[..., None, None, None]) * x0_ND + t_N[..., None, None, None] * x1_ND

    def ut_given_x1(self, xt_ND, x1_ND, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_ND - diff * xt_ND
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None, None, None]

    def get_samples(self, dataset, labels, n_samples):
        dataset_size = dataset.shape[0]
        selected_ind = np.random.randint(0, dataset_size - 1, n_samples)
        return dataset[selected_ind], labels[selected_ind]

    def train_2stage(self, dataset,labels, init_ckpt_path = None, epoches = configs.default_epoches, batch_size_N = configs.default_batchsize, lr = configs.default_lr):
        mymodel = self.get_untrained_model()
        if init_ckpt_path is not None:
            mymodel.load_state_dict(torch.load(init_ckpt_path, weights_only=True))
        optimizer = Adam(mymodel.parameters(), lr=lr)
        for j in tqdm.tqdm(range(epoches)):
            x1_ND, labels_N = self.get_samples(dataset, labels, batch_size_N)
            x0_ND = torch.randn_like(x1_ND, device=configs.device, dtype=torch.float32)

            t_N = torch.rand(batch_size_N, dtype=torch.float32, device=configs.device)
            xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
            ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
            # model_input = torch.cat([xt_ND, t_N[:, None]], dim=-1)
            vt_ND = mymodel(xt_ND, t_N[:, None])

            flow_loss = self.crieria(ut_ND, vt_ND)

            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

            if (j + 1) % configs.FMsave_every == 0 or j == 0:
                print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(mymodel.state_dict(), './saved_model/'+ 'FMworef_' + str(j + 1) + '.pth')

        return mymodel


    def train2_2stage(self, dataset, labels, basemodel_dict, epoches=configs.default_epoches,
                    batch_size_N=configs.default_batchsize_stage2,
                    lr=configs.default_lr):
        if configs.plot_loss == True:
            flow_loss_record = np.array([])
            adv_record = np.array([])
            constraint_loss_record = np.array([])


        if basemodel_dict is not None:
            ckpt1 = torch.load(basemodel_dict, map_location=configs.device)
            # self.policy.net.load_state_dict(ckpt1)
            self.policy.load_state_dict(ckpt1)

        optimizer = Adam([
            {'params': self.policy.net.parameters(), 'lr': lr},
            {'params': [self.policy.log_std], 'lr': 3e-6}
        ])
        #         for j in tqdm.tqdm(range(epoches)):
        # iter = 0
        with tqdm.tqdm(range(epoches), desc="") as pbar:
            for j in pbar:
                x1_N1TD, labels_N = self.get_samples(dataset, labels, batch_size_N)
                x0_N1TD = torch.randn_like(x1_N1TD, device=self.device, dtype=torch.float32)
                t_stage2_N = torch.ones(batch_size_N, dtype=torch.float32, device=self.device) * self.stage1_t
                xstage2_N1TD = self.sample_xt_given_x1_x0(x0_N1TD, x1_N1TD, t_stage2_N, sig_min=0)

                logPi_mat_NS = torch.ones(batch_size_N, self.RL_Steps_S, dtype=torch.float32,
                                          device=self.device)  # mychange zeros->ones
                cur_t_N = torch.zeros(batch_size_N, dtype=torch.float32, device=self.device) + self.stage1_t
                for i in range(self.RL_Steps_S):
                    vt_N1DD_grad, vt_N1DD, log_prob_N = self.policy.get_v(xstage2_N1TD, cur_t_N)
                    logPi_mat_NS[:, i] = log_prob_N
                    cur_t_N = cur_t_N + self.RL_step_width
                    xstage2_N1TD = xstage2_N1TD + vt_N1DD * self.RL_step_width

                with torch.no_grad():
                    preds = self.bb_model.predict(xstage2_N1TD)
                error_mat = preds != labels_N
                l2_norm = torch.norm(xstage2_N1TD - x1_N1TD, p=2, dim=(1, 2, 3))

                Cum_reward_mat_NS = (error_mat * self.adv_reward).unsqueeze(-1)
                constraint_loss = - (Cum_reward_mat_NS * logPi_mat_NS).mean()

                t_N = torch.randint(low=0, high=self.RL_Steps_S, size=(batch_size_N,),
                                    device=self.device) * self.RL_step_width + self.stage1_t
                xt_ND = self.sample_xt_given_x1_x0(x0_N1TD, x1_N1TD, t_N)
                ut_ND = self.ut_given_x1(xt_ND, x1_N1TD, t_N)
                # model_input = torch.cat([xt_ND, t_N[:, None]], dim=-1)
                vt_N1DD_grad, vt_N1DD, log_prob_N = self.policy.get_v(xt_ND, t_N)

                flow_loss = self.crieria(ut_ND, vt_N1DD_grad)
                # when computing flow_loss, remove the noise part from vt_N1DD_grad when the batchsize is small to avoid high variance

                loss = (flow_loss + constraint_loss)
                optimizer.zero_grad()

                loss.backward()

                optimizer.step()

                adv_success_num = torch.sum(error_mat)

                CL = constraint_loss
                # mean_std = torch.mean(st_N1TD)
                # def print_model_stats(model):
                # all_params = torch.cat([param.flatten() for param in mymodel.parameters() if param.requires_grad])
                # max_val = all_params.max().item()
                # min_val = all_params.min().item()
                pbar.set_postfix(
                    {'FL': '{:4f}'.format(flow_loss), 'adv confidence': '{:3f}'.format(adv_success_num),
                     'l2': '{:4f}'.format(l2_norm.mean().item()),
                     # 'mean_log': '{:4f}'.format(mean_log),
                     'CL': '{:4f}'.format(CL.item())})
                pbar.update(1)
                #                 print(j)/
                if configs.plot_loss == True:
                    if (j + 1) % configs.plot_loss_every == 0 or j == 0:
                        flow_loss_record = np.append(flow_loss_record, loss.detach().cpu().numpy())
                        adv_record = np.append(adv_record, adv_success_num.cpu().numpy())
                        # std_record = np.append(std_record, mean_std.detach().cpu().numpy())
                        constraint_loss_record = np.append(constraint_loss_record, CL.detach().cpu().numpy())

                if (j + 1) % configs.RLFMsave_every == 0 or j == 0:
                    #                 print(str(j) + ' Flow Loss: {:5f}'.format(loss))
                    #                 print(str(j) + ' Inside num: {:5f}'.format(torch.sum(constraint_mask)))
                    torch.save(self.policy.state_dict(),
                               './saved_model/' + configs.RLFMstage2_name + '_train_' + str(j + 1) + '.pth')
                    np.savez(
                        './saved_model/' + configs.RLFMstage2_name + '_' + 'train_record.npz',
                        flow_loss_record=flow_loss_record, adv_record=adv_record,
                        constraint_loss_record=constraint_loss_record)
                # print(j)
            return self.policy

    def RLFMsample(self, stage2_model, ref,
                   default_stage1t=None, default_RLstep_S=None):
        if default_stage1t is None:
            default_stage1t = self.stage1_t
        if default_RLstep_S is None:
            default_RLstep_S = self.RL_Steps_S
        # if default_RL_step_width is None:
        default_RL_step_width = (1 - default_stage1t) / (default_RLstep_S )

        x_prev = torch.randn(ref.shape[0], 3, 32, 32, dtype=torch.float32, device=self.device)
        t_tensor_N = default_stage1t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
        # t_tensor_N = tmpt * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32)
        x_prev = self.sample_xt_given_x1_x0(x_prev, ref, t_tensor_N,  sig_min=0)

        for i in range(default_RLstep_S):
            t = i * default_RL_step_width
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=self.device, dtype=torch.float32) + default_stage1t
            # input_ND = torch.cat((x_prev, t_tensor_N), dim=1)
            with torch.no_grad():
                vt_N1DD_grad, vt_N1DD, log_prob_N = stage2_model.get_v(x_prev, t_tensor_N, deterministic=True)
                # vt_N1DD = stage2_model(x_prev, t_tensor_N[:,None])
            x_prev = x_prev + vt_N1DD * default_RL_step_width
        return x_prev
