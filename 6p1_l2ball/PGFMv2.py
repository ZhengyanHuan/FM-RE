import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import configs
from dataset_MDM import check_inside_l2ball
import time

def get_linear_layer_block(num_layers, input_dim, hidden_dim, output_dim, activation=nn.SELU, dropout=0.0):

    layers = []
    current_dim = input_dim

    for i in range(num_layers):
        # Determine the output dimension for this layer
        if i == num_layers - 1:
            next_dim = output_dim  # Last layer should output the desired output_dim
        else:
            next_dim = hidden_dim  # Intermediate layers use hidden_dim

        # Add a linear layer
        layers.append(nn.Linear(current_dim, next_dim))

        # Add activation function and dropout if this is not the last layer
        if i < num_layers - 1:
            if activation is not None:
                layers.append(activation())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))

        # Update current dimension for the next layer
        current_dim = next_dim

    return nn.Sequential(*layers)

class mean_predictor(nn.Module):
    def __init__(self, d_model,t_layer_num = 2, x_layer_num = 3, tx_layer_num = 3, hidden_dim = 64):
        super(mean_predictor, self).__init__()
        self.t_net = get_linear_layer_block(t_layer_num, 1, hidden_dim, hidden_dim)
        self.x_net = get_linear_layer_block(x_layer_num, d_model, hidden_dim, hidden_dim)
        self.tx_net = get_linear_layer_block(tx_layer_num, hidden_dim, hidden_dim, d_model)


    def forward(self, x, t):
        combined_tx = self.t_net(t) + self.x_net(x)
        combined_tx = F.selu(combined_tx)
        mean = self.tx_net(combined_tx)
        return mean

class PolicyNet(nn.Module):
    def __init__(self, d_model,t_layer_num = 2, x_layer_num = 3, tx_layer_num = 3, hidden_dim = 64):
        super(PolicyNet, self).__init__()
        self.net = mean_predictor(d_model, t_layer_num, x_layer_num,tx_layer_num, hidden_dim)
        self.log_std = nn.Parameter(torch.zeros(d_model))

    def forward(self, x, t):
        mean = self.net(x, t)
        std = torch.exp(self.log_std)+1e-5

        return mean, std



    def get_v(self, cur_state_ND, cur_t_N, deterministic=False):
        # model_input = torch.cat([cur_state_ND, cur_t_N[:, None]], dim=-1)
        mean, std = self.forward(cur_state_ND, cur_t_N[:, None])
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
            # print(log_prob_N)
        return v_ND_grad, v_ND, log_prob_N



class PGFM:

    def __init__(self, sig_min: float = configs.default_sig_min, stage1_t=configs.default_stage1_t,
                 RL_Steps_S=configs.default_RL_Steps_S, d_model=configs.d_model
                 , device=configs.device, default_generation_step = configs.default_generation_step,
                 constraint_reward=configs.default_constraint_reward) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.crieria = nn.MSELoss()
        self.stage1_t = stage1_t
        self.RL_Steps_S = RL_Steps_S
        self.d_model = d_model
        self.device = device
        self.RL_step_width = (1 - self.stage1_t) / (self.RL_Steps_S )

        self.default_generation_step = default_generation_step
        self.constraint_reward = constraint_reward
        self.check_inside_l2ball = check_inside_l2ball
        self.policy = PolicyNet(d_model = configs.d_model).to(self.device)

    def get_untrained_model(self):

        return mean_predictor(self.d_model).to(configs.device)

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

    def get_Curwd(self, reward_mat_NS_d):
        fliped_reward_mat_MS = torch.flip(reward_mat_NS_d, dims=[1])
        fliped_Cum_reward_mat_NS = torch.cumsum(fliped_reward_mat_MS, dim=-1)
        Cum_reward_mat_NS = torch.flip(fliped_Cum_reward_mat_NS, dims=[1])
        return Cum_reward_mat_NS

    def get_terminal_rwd(self, cur_state_ND):
        valid_mask = self.check_inside_l2ball(cur_state_ND)
        in_num = torch.sum(valid_mask).item()
        return valid_mask * self.constraint_reward, in_num

    def ut_given_x1(self, xt_ND, x1_ND, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_ND - diff * xt_ND
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None]

    def train2_2stage(self, dataset,ckpt_path, epoches=configs.default_epoches, batch_size_N=configs.default_batchsize_stage2,
                     lr=configs.default_lr):

        if configs.plot_loss == True:
            loss_record = np.array([])
            in_prob_record = np.array([])


        # if ckpt_path is not None:
        ckpt1 = torch.load(ckpt_path, map_location=configs.device, weights_only= True)
        self.policy.net.load_state_dict(ckpt1)
        stage1_model = self.get_untrained_model()
        stage1_model.load_state_dict(ckpt1)

        # optimizer = Adam(self.policy.parameters(), lr=lr)
        optimizer = Adam([
            {'params': self.policy.net.parameters(), 'lr': lr},
            {'params': [self.policy.log_std], 'lr': 1e-4}
        ])

        with (tqdm.tqdm(range(epoches), desc="") as pbar):
            for j in pbar:
                x1_ND = self.get_samples(dataset, batch_size_N)
                x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)
                # t_stage2_N = torch.ones(batch_size_N, dtype=torch.float32, device=self.device) * self.stage1_t
                xstage2_ND = self.get_xt0(stage1_model, x0_ND)

                cur_t_N = torch.zeros(batch_size_N, dtype=torch.float32,
                                      device=self.device) * self.RL_step_width + self.stage1_t

                logPi_mat_NS = torch.ones(batch_size_N, self.RL_Steps_S, dtype=torch.float32,
                                          device=self.device)
                for i in range(self.RL_Steps_S):
                    v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xstage2_ND, cur_t_N)

                    logPi_mat_NS[:, i] = log_prob_N

                    cur_t_N = cur_t_N + self.RL_step_width
                    xstage2_ND = xstage2_ND + v_ND * self.RL_step_width

                terminal_rwd, in_num = self.get_terminal_rwd(xstage2_ND)
                Cum_reward_mat_NS = terminal_rwd.unsqueeze(-1)
                constraintloss = (- logPi_mat_NS * Cum_reward_mat_NS).mean()

                t_N = torch.randint(low=0, high=self.RL_Steps_S, size=(batch_size_N,), device=self.device) * self.RL_step_width + self.stage1_t
                xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
                ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
                v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xt_ND, t_N)
                flow_loss = self.crieria(ut_ND, v_ND_grad)

                loss = (flow_loss + constraintloss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {'flow_loss': '{:5f}'.format(torch.mean(flow_loss).item()),
                     'constraintloss': '{:5f}'.format(torch.mean(constraintloss).item()),
                     'logp': '{:5f}'.format(torch.mean(logPi_mat_NS).item()),
                     'inum': '{:5f}'.format(in_num)})
                pbar.update(1)
                #                 print(j)/
                if configs.plot_loss == True:
                    if (j + 1) % configs.plot_loss_every == 0 or j == 0:
                        loss_record = np.append(loss_record, loss.detach().cpu().numpy())
                        in_prob_record = np.append(in_prob_record,
                                                   in_num / configs.default_batchsize_stage2)

                if (j + 1) % configs.RLFMsave_every == 0 or j == 0:
                    #                 print(str(j) + ' Flow Loss: {:5f}'.format(loss))
                    #                 print(str(j) + ' Innum: {:5f}'.format(torch.sum(constraint_mask)))
                    torch.save(self.policy.state_dict(),
                               './saved_model/' + configs.RLFMstage2_name + '_' + str(j + 1) + '.pth')
                    np.savez(
                        './saved_model/' + configs.RLFMstage2_name + '_' + 'train_record.npz',
                        loss_record=loss_record, in_prob_record=in_prob_record)



    def train2_2stage_batch(self, dataset,ckpt_path, save_name, epoches=configs.default_epoches, batch_size_N=configs.default_batchsize_stage2,
                     lr=configs.default_lr):

        print(save_name)
        # if ckpt_path is not None:
        ckpt1 = torch.load(ckpt_path, map_location=configs.device, weights_only= True)
        self.policy.net.load_state_dict(ckpt1)
        stage1_model = self.get_untrained_model()
        stage1_model.load_state_dict(ckpt1)

        # optimizer = Adam(self.policy.parameters(), lr=lr)
        optimizer = Adam([
            {'params': self.policy.net.parameters(), 'lr': lr},
            {'params': [self.policy.log_std], 'lr': 1e-4}
        ])

        # with (tqdm.tqdm(range(epoches), desc="") as pbar):
        start_time = time.time()
        for j in range(epoches):
            x1_ND = self.get_samples(dataset, batch_size_N)
            x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)
            xstage2_ND = self.get_xt0(stage1_model, x0_ND)

            cur_t_N = torch.zeros(batch_size_N, dtype=torch.float32,
                                  device=self.device) * self.RL_step_width + self.stage1_t

            logPi_mat_NS = torch.ones(batch_size_N, self.RL_Steps_S, dtype=torch.float32,
                                      device=self.device)
            for i in range(self.RL_Steps_S):
                v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xstage2_ND, cur_t_N)

                logPi_mat_NS[:, i] = log_prob_N

                cur_t_N = cur_t_N + self.RL_step_width
                xstage2_ND = xstage2_ND + v_ND * self.RL_step_width

            terminal_rwd, in_num = self.get_terminal_rwd(xstage2_ND)
            Cum_reward_mat_NS = terminal_rwd.unsqueeze(-1)
            constraintloss = (- logPi_mat_NS * Cum_reward_mat_NS).mean()

            t_N = torch.randint(low=0, high=self.RL_Steps_S, size=(batch_size_N,), device=self.device) * self.RL_step_width + self.stage1_t
            xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
            ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
            v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xt_ND, t_N)
            flow_loss = self.crieria(ut_ND, v_ND_grad)

            loss = (flow_loss + constraintloss)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        end_time = time.time()
        training_time = int(end_time - start_time)
        torch.save(self.policy.state_dict(),
                   './saved_model_2/' + save_name  + '.pth')

        text_to_append = '\n t0: '+str(self.stage1_t) +'  lambda: '+str(self.constraint_reward) + '  training_time: '+str(training_time)
        with open('./saved_model_2/training_time_record', 'a') as f:
            f.write(text_to_append)



    def check_distance_l2ball(self, mat_Ndim):
        out = torch.norm(mat_Ndim, dim=-1) - 1
        out[out < 0] = 0
        return out

    def get_terminal_feedback(self, cur_state_ND):
        valid_mask = self.check_inside_l2ball(cur_state_ND.detach())
        in_num = torch.sum(valid_mask).item()
        terminal_feedback = self.check_distance_l2ball(cur_state_ND)
        return terminal_feedback.mean() * self.constraint_reward, in_num

    def train2_2stage_w_diff_distance(self, dataset,ckpt_path, epoches=configs.default_epoches, batch_size_N=configs.default_batchsize_stage2,
                     lr=configs.default_lr):

        if configs.plot_loss == True:
            loss_record = np.array([])
            in_prob_record = np.array([])


        # if ckpt_path is not None:
        ckpt1 = torch.load(ckpt_path, map_location=configs.device, weights_only= True)
        stage1_model = self.get_untrained_model()
        stage1_model.load_state_dict(ckpt1)
        stage2_model = self.get_untrained_model()
        stage2_model.load_state_dict(ckpt1)

        optimizer = Adam(stage2_model.parameters(), lr=lr)

        with (tqdm.tqdm(range(epoches), desc="") as pbar):
            for j in pbar:
                x1_ND = self.get_samples(dataset, batch_size_N)
                x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)
                # t_stage2_N = torch.ones(batch_size_N, dtype=torch.float32, device=self.device) * self.stage1_t
                xstage2_ND = self.get_xt0(stage1_model, x0_ND)

                cur_t_N = torch.zeros(batch_size_N, dtype=torch.float32,
                                      device=self.device) * self.RL_step_width + self.stage1_t


                for i in range(self.RL_Steps_S):
                    v_ND_grad = stage2_model(xstage2_ND.detach(), cur_t_N[:, None])
                    cur_t_N = cur_t_N + self.RL_step_width
                    xstage2_ND = xstage2_ND + v_ND_grad * self.RL_step_width

                terminal_feedback, in_num = self.get_terminal_feedback(xstage2_ND)


                t_N = torch.randint(low=0, high=self.RL_Steps_S, size=(batch_size_N,), device=self.device) * self.RL_step_width + self.stage1_t
                xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
                ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
                v_ND_grad= stage2_model(xt_ND, t_N[:, None])
                flow_loss = self.crieria(ut_ND, v_ND_grad)

                loss = (flow_loss + terminal_feedback)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {'flow_loss': '{:5f}'.format(torch.mean(flow_loss).item()),
                     'constraintloss': '{:5f}'.format(torch.mean(terminal_feedback).item()),
                     # 'logp': '{:5f}'.format(torch.mean(logPi_mat_NS).item()),
                     'inum': '{:5f}'.format(in_num)})
                pbar.update(1)
                #                 print(j)/
                if configs.plot_loss == True:
                    if (j + 1) % configs.plot_loss_every == 0 or j == 0:
                        loss_record = np.append(loss_record, loss.detach().cpu().numpy())
                        in_prob_record = np.append(in_prob_record,
                                                   in_num / configs.default_batchsize_stage2)

                if (j + 1) % configs.RLFMsave_every == 0 or j == 0:
                    #                 print(str(j) + ' Flow Loss: {:5f}'.format(loss))
                    #                 print(str(j) + ' Inside num: {:5f}'.format(torch.sum(constraint_mask)))
                    torch.save(stage2_model.state_dict(),
                               './saved_model/' + configs.RLFMstage2_name + '_diffd_' + str(j + 1) + '.pth')
                    np.savez(
                        './saved_model/' + configs.RLFMstage2_name + '_' + 'diffd_train_record.npz',
                        loss_record=loss_record, in_prob_record=in_prob_record)


    def get_xt0(self, stage1_model, x0_ND):
        x_prev = x0_ND
        delta_t = 1 / self.default_generation_step * self.stage1_t
        t_tensor_N = torch.zeros(x_prev.shape[0], device=configs.device, dtype=torch.float32)
        for i in range(self.default_generation_step):
            with torch.no_grad():
                z = stage1_model(x_prev, t_tensor_N[:, None])
            x_prev = x_prev + z * 1 / self.default_generation_step * self.stage1_t
            t_tensor_N = t_tensor_N + delta_t

        return x_prev

    def PGFMsample_train2(self, stage1_model, stage2_model, batch_size, mode = "policy"):
        x_prev = torch.randn(batch_size, configs.d_model, dtype=torch.float32, device=configs.device)
        for i in range(self.default_generation_step):
            t = i / self.default_generation_step * self.stage1_t
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:, None]), dim=1)
            with torch.no_grad():
                z = stage1_model(x_prev, t_tensor_N[:, None])
            x_prev = x_prev + z * 1 / self.default_generation_step * self.stage1_t

        # RL_step_width = (1 - stage1_t) / (RL_Steps_S + 1)
        for i in range(self.RL_Steps_S):
            t = i * self.RL_step_width
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32) + self.stage1_t
            with torch.no_grad():
                if  mode == "policy":
                    v_ND_grad, v_ND, log_prob_N = stage2_model.get_v(x_prev, t_tensor_N, deterministic=True)
                elif  mode == "deterministic":
                    v_ND = stage2_model(x_prev, t_tensor_N[:,None])
                # print(log_prob_N)

            x_prev = x_prev + v_ND * self.RL_step_width
        return x_prev



















