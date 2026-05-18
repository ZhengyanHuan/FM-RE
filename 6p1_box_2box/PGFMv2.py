import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from torch.backends.cudnn import deterministic

import configs
import tqdm

# def get_untrained_model(self):
#     # torch.manual_seed(42)
#     # REF https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/models.py
#     return nn.Sequential(
#         nn.Linear(3, 64),
#         nn.SELU(),
#         nn.LazyLinear(out_features=64),
#         nn.SELU(),
#         nn.LazyLinear(64),
#         nn.SELU(),
#         nn.LazyLinear(out_features=64),
#         nn.SELU(),
#         nn.LazyLinear(64),
#         nn.SELU(),
#         nn.LazyLinear(out_features=self.output_dim)
#     ).to(self.device)



class PolicyNet(nn.Module):
    def __init__(self, STATE_DIM, ACTION_DIM):
        super().__init__()
        self.net = nn.Sequential(
                nn.Linear(STATE_DIM, 64),
                nn.SELU(),
                nn.LazyLinear(out_features=64),
                nn.SELU(),
                nn.LazyLinear(64),
                nn.SELU(),
                nn.LazyLinear(out_features=64),
                nn.SELU(),
                nn.LazyLinear(64),
                nn.SELU(),
                nn.LazyLinear(out_features=ACTION_DIM)
            )
        self.log_std = nn.Parameter(torch.zeros(ACTION_DIM)) #control the initial exploration range


    def forward(self, state):
        mean = self.net(state)
        std = torch.sigmoid(self.log_std)+1e-5
        # std = torch.exp(self.log_std)
        return mean, std

    def get_v(self, cur_state_ND, cur_t_N, deterministic=False):
        model_input = torch.cat([cur_state_ND, cur_t_N[:, None]], dim=-1)
        mean, std = self.forward(model_input)
        # print(mean, std)
        if deterministic:
            v_ND_grad = mean
            v_ND = v_ND_grad.detach()
            log_prob_N = torch.zeros_like(mean).sum(-1)
        else:
            dist = torch.distributions.Normal(mean, std)
            # v_ND = dist.sample()
            v_ND_grad = mean + std * torch.randn_like(mean)
            v_ND = v_ND_grad.detach()
            log_prob_N = dist.log_prob(v_ND).sum(-1)
        return v_ND_grad, v_ND, log_prob_N

class PGFM:
    def __init__(self, sig_min: float = configs.default_sig_min, stage1_t=configs.default_stage1_t,
                 RL_Steps_S=configs.default_RL_Steps_S, output_dim=configs.output_dim
                 , device=configs.device, box_center=configs.uniform_center, box_bound=configs.bound,
                 default_generation_step=configs.default_generation_step,
                 constraint_reward=configs.default_constraint_reward, dataset = configs.dataset) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.crieria = nn.MSELoss()
        self.stage1_t = stage1_t
        self.RL_Steps_S = RL_Steps_S
        self.output_dim = output_dim
        self.device = device
        self.RL_step_width = (1 - self.stage1_t) / (self.RL_Steps_S )
        self.box_center = box_center
        self.box_bound = box_bound
        self.default_generation_step = default_generation_step
        self.constraint_reward = constraint_reward
        self.policy = PolicyNet(self.output_dim + 1, self.output_dim).to(self.device)
        if dataset != "uniform_box":
            self.default_check_inside = self.check_inside2
        else:
            self.default_check_inside = self.check_inside

    def check_inside(self, cur_state_ND):
        cur_state_center_ND = cur_state_ND - self.box_center
        return (cur_state_center_ND[:, 0] < self.box_bound) * (cur_state_center_ND[:, 0] > -self.box_bound) * \
               (cur_state_center_ND[:, 1] < self.box_bound) * (cur_state_center_ND[:, 1] > -self.box_bound)

    def check_inside2(self, cur_state_ND):
        cur_state_center_ND = torch.abs(cur_state_ND) - self.box_center
        return (cur_state_center_ND[:, 0] < self.box_bound) * (cur_state_center_ND[:, 0] > -self.box_bound) * \
               (cur_state_center_ND[:, 1] < self.box_bound) * (cur_state_center_ND[:, 1] > -self.box_bound)


    def sample_xt_given_x1_x0(self, x0_ND: torch.Tensor, x1_ND: torch.Tensor, t_N: torch.Tensor, sig_min = None):
        # N, D = x1_ND.shape
        if sig_min is None:
            std1 = self.sig_min
        else:
            std1 = sig_min
        return (1 - (1 - std1) * t_N[..., None]) * x0_ND + t_N[..., None] * x1_ND

    def get_samples(self, dataset, n_samples):
        dataset_size = dataset.shape[0]
        selected_ind = np.random.randint(0, dataset_size - 1, n_samples)
        return dataset[selected_ind]

    def get_untrained_model(self):
        # torch.manual_seed(42)
        # REF https://github.com/atong01/conditional-flow-matching/blob/main/torchcfm/models/models.py
        return nn.Sequential(
            nn.Linear(3, 64),
            nn.SELU(),
            nn.LazyLinear(out_features=64),
            nn.SELU(),
            nn.LazyLinear(64),
            nn.SELU(),
            nn.LazyLinear(out_features=64),
            nn.SELU(),
            nn.LazyLinear(64),
            nn.SELU(),
            nn.LazyLinear(out_features=self.output_dim)
        ).to(self.device)

    def ut_given_x1(self, xt_ND, x1_ND, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_ND - diff * xt_ND
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None]

    def get_terminal_rwd(self, cur_state_ND):
        valid_mask = self.default_check_inside(cur_state_ND)
        in_num = torch.sum(valid_mask).item()
        return valid_mask * self.constraint_reward, in_num

    def train2_2stage(self, dataset,ckpt_path, epoches=configs.default_epoches, batch_size_N=configs.default_batchsize_stage2,
                     lr=configs.default_lr):
        if configs.plot_loss == True:
            loss_record = np.array([])
            in_prob_record = np.array([])

        ckpt1 = torch.load(ckpt_path, map_location=configs.device, weights_only= True)
        self.policy.net.load_state_dict(ckpt1)
        stage1_model = self.get_untrained_model()
        stage1_model.load_state_dict(ckpt1)

        # optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        optimizer = optim.Adam([
            {'params': self.policy.net.parameters(), 'lr': lr},
            {'params': [self.policy.log_std], 'lr': 1e-4}
        ])
        with (tqdm.tqdm(range(epoches), desc="") as pbar):
            for j in pbar:
                ######################## constraint
                x1_ND = self.get_samples(dataset, batch_size_N)
                x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)
                # t_stage2_N = torch.ones(batch_size_N, dtype=torch.float32, device=self.device) * self.stage1_t
                xstage2_ND = self.get_xt0(stage1_model, x0_ND)

                cur_t_N = torch.zeros(batch_size_N, dtype=torch.float32, device=self.device) * self.RL_step_width + self.stage1_t

                logPi_mat_NS = torch.ones(batch_size_N, self.RL_Steps_S, dtype=torch.float32,
                                          device=self.device)
                for i in range(self.RL_Steps_S):
                    v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xstage2_ND, cur_t_N)
                    logPi_mat_NS[:, i] = log_prob_N
                    cur_t_N = cur_t_N + self.RL_step_width
                    xstage2_ND = xstage2_ND + v_ND * self.RL_step_width

                terminal_rwd, in_num = self.get_terminal_rwd(xstage2_ND)
                Cum_reward_mat_NS = terminal_rwd.unsqueeze(-1)
                constraintloss = (- logPi_mat_NS * Cum_reward_mat_NS)
                ######################
                ################### FM

                t_N = torch.randint(low=0, high=self.RL_Steps_S, size=(batch_size_N,), device=self.device) * self.RL_step_width + self.stage1_t
                xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
                ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
                v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xt_ND, t_N)
                flow_loss = self.crieria(ut_ND, v_ND_grad)
                #when computing flow_loss, remove the noise part from v_ND_grad when the batchsize is small to avoid high variance

                #############################


                loss = (flow_loss+constraintloss).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {'flow_loss': '{:5f}'.format(torch.mean(flow_loss).item()),
                     'constraintloss': '{:5f}'.format(torch.mean(constraintloss).item()),
                     'logp': '{:5f}'.format(torch.mean(logPi_mat_NS).item()),
                     'inside num': '{:5f}'.format(in_num)})
                pbar.update(1)
                #                 print(j)/
                if configs.plot_loss == True:
                    if (j + 1) % configs.plot_loss_every == 0 or j == 0:
                        loss_record = np.append(loss_record, loss.detach().cpu().numpy())
                        in_prob_record = np.append(in_prob_record,
                                                   in_num / configs.default_batchsize_stage2)

                if (j + 1) % configs.save_every == 0 or j == 0:
                    #                 print(str(j) + ' Flow Loss: {:5f}'.format(loss))
                    #                 print(str(j) + ' Inside num: {:5f}'.format(torch.sum(constraint_mask)))
                    torch.save(self.policy.state_dict(),
                               './saved_model/' + configs.RLFMstage2_name + '_' + str(j + 1) + '.pth')
                    np.savez(
                        './saved_model/' + configs.RLFMstage2_name + '_' + 'train_record.npz',
                        loss_record=loss_record, in_prob_record=in_prob_record)


    def train2_2stage_distance(self, dataset,ckpt_path, epoches=configs.default_epoches, batch_size_N=configs.default_batchsize_stage2,
                     lr=configs.default_lr):
        if configs.plot_loss == True:
            loss_record = np.array([])
            in_prob_record = np.array([])

        ckpt1 = torch.load(ckpt_path, map_location=configs.device, weights_only= True)
        self.policy.net.load_state_dict(ckpt1)
        stage1_model = self.get_untrained_model()
        stage1_model.load_state_dict(ckpt1)

        # optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        optimizer = optim.Adam([
            {'params': self.policy.net.parameters(), 'lr': lr},
            {'params': [self.policy.log_std], 'lr': 1e-4}
        ])
        with (tqdm.tqdm(range(epoches), desc="") as pbar):
            for j in pbar:
                ######################## constraint
                x1_ND = self.get_samples(dataset, batch_size_N)
                x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)
                # t_stage2_N = torch.ones(batch_size_N, dtype=torch.float32, device=self.device) * self.stage1_t
                xstage2_ND = self.get_xt0(stage1_model, x0_ND)

                cur_t_N = torch.zeros(batch_size_N, dtype=torch.float32, device=self.device) * self.RL_step_width + self.stage1_t

                # logPi_mat_NS = torch.ones(batch_size_N, self.RL_Steps_S, dtype=torch.float32,
                #                           device=self.device)
                for i in range(self.RL_Steps_S):
                    model_inp = torch.cat([xstage2_ND.detach(), cur_t_N[:, None]], dim=-1)
                    v_ND_grad = self.policy.net(model_inp)
                    # logPi_mat_NS[:, i] = log_prob_N
                    cur_t_N = cur_t_N + self.RL_step_width
                    xstage2_ND = xstage2_ND + v_ND_grad * self.RL_step_width

                terminal_rwd, in_num = self.get_terminal_rwd(xstage2_ND)
                # Cum_reward_mat_NS = terminal_rwd.unsqueeze(-1)
                constraintloss = configs.distance(xstage2_ND)*(terminal_rwd==0)
                ######################
                ################### FM

                t_N = torch.randint(low=0, high=self.RL_Steps_S, size=(batch_size_N,), device=self.device) * self.RL_step_width + self.stage1_t
                xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
                ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
                model_inp = torch.cat([xt_ND, t_N[:, None]], dim=-1)
                v_ND_grad = self.policy.net(model_inp)
                flow_loss = self.crieria(ut_ND, v_ND_grad)
                #############################


                loss = (flow_loss).mean()+constraintloss.mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {'flow_loss': '{:5f}'.format(torch.mean(flow_loss).item()),
                     'constraintloss': '{:5f}'.format(torch.mean(constraintloss).item()),
                     # 'logp': '{:5f}'.format(torch.mean(logPi_mat_NS).item()),
                     'inside num': '{:5f}'.format(in_num)})
                pbar.update(1)
                #                 print(j)/
                if configs.plot_loss == True:
                    if (j + 1) % configs.plot_loss_every == 0 or j == 0:
                        loss_record = np.append(loss_record, loss.detach().cpu().numpy())
                        in_prob_record = np.append(in_prob_record,
                                                   in_num / configs.default_batchsize_stage2)

                if (j + 1) % configs.save_every == 0 or j == 0:
                    #                 print(str(j) + ' Flow Loss: {:5f}'.format(loss))
                    #                 print(str(j) + ' Inside num: {:5f}'.format(torch.sum(constraint_mask)))
                    torch.save(self.policy.state_dict(),
                               './saved_model/' + configs.RLFMstage2_name + '_' + str(j + 1) + '.pth')
                    np.savez(
                        './saved_model/' + configs.RLFMstage2_name + '_' + 'train_record.npz',
                        loss_record=loss_record, in_prob_record=in_prob_record)



    def get_xt0(self, stage1_model, x0_ND):
        x_prev = x0_ND
        delta_t = 1 / self.default_generation_step * self.stage1_t
        t_tensor_N = torch.zeros(x_prev.shape[0], device=configs.device, dtype=torch.float32)
        for i in range(self.default_generation_step):
            with torch.no_grad():
                z = stage1_model(torch.cat([x_prev, t_tensor_N[:, None]], dim=-1))
            x_prev = x_prev + z * 1 / self.default_generation_step * self.stage1_t
            t_tensor_N = t_tensor_N + delta_t

        return x_prev
            
    def PGFMsample(self, stage1_model, stage2_model, batch_size):
        x_prev = torch.randn(batch_size, 2, dtype=torch.float32, device=configs.device)
        for i in range(configs.default_generation_step):
            t = i / configs.default_generation_step * self.stage1_t
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32)
            input_ND = torch.cat((x_prev, t_tensor_N[:, None]), dim=1)
            with torch.no_grad():
                z = stage1_model(input_ND)
            x_prev = x_prev + z * 1 / configs.default_generation_step * self.stage1_t

        # RL_step_width = (1 - stage1_t) / (RL_Steps_S + 1)
        for i in range(self.RL_Steps_S):
            t = i * self.RL_step_width
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32) + self.stage1_t
            with torch.no_grad():
                v_ND_grad, v_ND, log_prob_N = stage2_model.get_v(x_prev, t_tensor_N, deterministic=True)

            x_prev = x_prev + v_ND * self.RL_step_width
        return x_prev