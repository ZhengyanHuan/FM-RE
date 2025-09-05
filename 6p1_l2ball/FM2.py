import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import configs
from dataset_MDM import check_inside_l2ball




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


class FM2:

    def __init__(self, sig_min: float = configs.default_sig_min, stage1_t=configs.default_stage1_t,
                 RL_Steps_S=configs.default_RL_Steps_S, d_model=configs.d_model
                 , device=configs.device,
                 constraint_reward=configs.default_constraint_reward) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.crieria = nn.MSELoss()
        self.stage1_t = stage1_t
        self.RL_Steps_S = RL_Steps_S
        self.d_model = d_model
        self.device = device
        self.RL_step_width = (1 - self.stage1_t) / (self.RL_Steps_S )

        self.constraint_reward = constraint_reward
        self.check_inside_l2ball = check_inside_l2ball
        self.policy = mean_predictor(d_model = configs.d_model).to(self.device)

    def check_distance_l2ball(self, mat_Ndim):
        out = torch.norm(mat_Ndim, dim=-1) - 1
        out[out < 0] = 0
        return out
    
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

    # def get_Curwd(self, reward_mat_NS_d):
    #     fliped_reward_mat_MS = torch.flip(reward_mat_NS_d, dims=[1])
    #     fliped_Cum_reward_mat_NS = torch.cumsum(fliped_reward_mat_MS, dim=-1)
    #     Cum_reward_mat_NS = torch.flip(fliped_Cum_reward_mat_NS, dims=[1])
    #     return Cum_reward_mat_NS

    def get_terminal_rwd(self, cur_state_ND):
        valid_mask = self.check_inside_l2ball(cur_state_ND.detach())
        in_num = torch.sum(valid_mask).item()
        terminal_rwd = self.check_distance_l2ball(cur_state_ND)
        return terminal_rwd * self.constraint_reward, in_num

    def train2_2stage(self, dataset,ckpt_path = None, epoches=configs.default_epoches, batch_size_N=configs.default_batchsize_stage2,
                     lr=configs.default_lr):

        if configs.plot_loss == True:
            loss_record = np.array([])
            in_prob_record = np.array([])

        if ckpt_path is not None:
            ckpt1 = torch.load(ckpt_path, map_location=configs.device, weights_only= True)
            self.policy.load_state_dict(ckpt1)

        optimizer = Adam(self.policy.parameters(), lr=lr)
        with (tqdm.tqdm(range(epoches), desc="Training") as pbar):
            for j in pbar:
                x1_ND = self.get_samples(dataset, batch_size_N)
                x0_ND = torch.randn_like(x1_ND, device=self.device, dtype=torch.float32)
                t_stage2_N = torch.ones(batch_size_N, dtype=torch.float32, device=self.device) * self.stage1_t
                xstage2_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_stage2_N, sig_min=0)

                cur_t_N = torch.zeros(batch_size_N, dtype=torch.float32,
                                      device=self.device) * self.RL_step_width + self.stage1_t

                reward_mat_NS = torch.zeros(batch_size_N, self.RL_Steps_S, dtype=torch.float32, device=self.device)
                # logPi_mat_NS = torch.ones(batch_size_N, self.RL_Steps_S, dtype=torch.float32,
                #                           device=self.device)
                for i in range(self.RL_Steps_S):
                    # v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xstage2_ND, cur_t_N)
                    v_ND_grad = self.policy(xstage2_ND.detach(), cur_t_N[:, None])
                    # logPi_mat_NS[:, i] = log_prob_N

                    cur_t_N = cur_t_N + self.RL_step_width
                    phi_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, cur_t_N, sig_min=0)
                    reward_mat_NS[:, i] = (
                        - torch.norm(xstage2_ND.detach() + v_ND_grad * self.RL_step_width - phi_ND, dim=-1) ** 2)

                    xstage2_ND = xstage2_ND + v_ND_grad * self.RL_step_width

                terminal_rwd, in_num = self.get_terminal_rwd(xstage2_ND)
                reward_mat_NS[:, -1] = reward_mat_NS[:, -1] - terminal_rwd
                # reward_mat_NS_d = reward_mat_NS.detach()
                # Cum_reward_mat_NS = self.get_Curwd(reward_mat_NS_d)
                # loss1 = (- logPi_mat_NS * Cum_reward_mat_NS)
                loss = -reward_mat_NS.mean()

                # loss = (loss1 + loss2).mean()
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {'Loss1': '{:5f}'.format(torch.mean(loss).item()),
                     # 'logp': '{:5f}'.format(torch.mean(logPi_mat_NS).item()),
                     'inside num': '{:5f}'.format(in_num)})
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
                    torch.save(self.policy.state_dict(),
                               './saved_model/' + configs.RLFMstage2_name + '_' + str(j + 1) + '.pth')
                    np.savez(
                        './saved_model/' + configs.RLFMstage2_name + '_' + 'train_record.npz',
                        loss_record=loss_record, in_prob_record=in_prob_record)

    def PGFMsample_train2(self, stage1_model, stage2_model, batch_size):
        x_prev = torch.randn(batch_size, configs.d_model, dtype=torch.float32, device=configs.device)
        for i in range(configs.default_generation_step):
            t = i / configs.default_generation_step * self.stage1_t
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32)
            # input_ND = torch.cat((x_prev, t_tensor_N[:, None]), dim=1)
            with torch.no_grad():
                z = stage1_model(x_prev, t_tensor_N[:, None])
            x_prev = x_prev + z * 1 / configs.default_generation_step * self.stage1_t

        # RL_step_width = (1 - stage1_t) / (RL_Steps_S + 1)
        for i in range(self.RL_Steps_S):
            t = i * self.RL_step_width
            t_tensor_N = t * torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32) + self.stage1_t
            with torch.no_grad():
                # v_ND_grad, v_ND, log_prob_N = stage2_model.get_v(x_prev, t_tensor_N, deterministic=True)
                v_ND = stage2_model(x_prev, t_tensor_N[:, None])

            x_prev = x_prev + v_ND * self.RL_step_width
        return x_prev



















