import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import configs


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
        self.log_std = nn.Parameter(torch.zeros(d_model)) #control the initial exploration range

    def forward(self, x, t):
        mean = self.net(x, t)
        std = (torch.exp(self.log_std)+1e-5)/50 # control the exploration
        # print(mean.shape, std.shape)
        return mean, std



    def get_v(self, cur_state_ND, cur_t_N, deterministic=False):
        # model_input = torch.cat([cur_state_ND, cur_t_N[:, None]], dim=-1)
        mean, std = self.forward(cur_state_ND, cur_t_N[:, None])
        # print(std)
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
            # print(log_prob_N)
        return v_ND_grad, v_ND, log_prob_N



class PGFM:

    def __init__(self, sig_min: float = configs.default_sig_min, stage1_t=configs.default_stage1_t,
                 RL_Steps_S=configs.default_RL_Steps_S, d_model=configs.d_model
                 , device=configs.device,
                 constraint_reward=configs.default_constraint_reward, coeffs = None) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.crieria = nn.MSELoss()
        self.stage1_t = stage1_t
        self.RL_Steps_S = RL_Steps_S
        self.d_model = d_model
        self.device = device
        self.tolerance = 5e-4
        self.RL_step_width = (1 - self.stage1_t) / (self.RL_Steps_S )

        self.constraint_reward = constraint_reward
        if coeffs is not None:
            self.coeffs = coeffs
        else:
            self.coeffs = torch.ones(self.d_model + 1).to(self.device)
            self.coeffs[-1] = 10
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

    def get_terminal_rwd(self, points_Nd):
        normal = self.coeffs[:-1]  # Extract normal vector
        numerator = torch.abs(points_Nd @ normal + self.coeffs[-1])
        denominator = torch.norm(normal)
        distance = numerator / denominator
        mean_distance = torch.mean(distance).detach().cpu() # only for display purpose

        reward = torch.zeros_like(distance)
        reward[distance <= self.tolerance] = self.constraint_reward
        reward[distance > self.tolerance] = 0
        return reward, mean_distance

    def ut_given_x1(self, xt_ND, x1_ND, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_ND - diff * xt_ND
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None]

    def train2_2stage(self, dataset,ckpt_path , epoches=configs.default_epoches, batch_size_N=configs.default_batchsize_stage2,
                     lr=configs.default_lr):

        if configs.plot_loss == True:
            loss_record = np.array([])
            avg_distance_record = np.array([])

        ckpt1 = torch.load(ckpt_path, map_location=configs.device, weights_only= True)
        self.policy.net.load_state_dict(ckpt1)
        stage1_model = self.get_untrained_model()
        stage1_model.load_state_dict(ckpt1)

        # optimizer = Adam(self.policy.parameters(), lr=lr)
        optimizer = Adam([
            {'params': self.policy.net.parameters(), 'lr': lr},
            {'params': [self.policy.log_std], 'lr': 1e-4}
        ])

        with (tqdm.tqdm(range(epoches), desc="Training") as pbar):
            for j in pbar:
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

                terminal_rwd, mean_distance = self.get_terminal_rwd(xstage2_ND)
                Cum_reward_mat_NS = terminal_rwd.unsqueeze(-1)
                constraintloss = (- logPi_mat_NS * Cum_reward_mat_NS).mean()


                t_N = torch.randint(low=0, high=self.RL_Steps_S, size=(batch_size_N,), device=self.device) * self.RL_step_width + self.stage1_t
                xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
                ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
                v_ND_grad, v_ND, log_prob_N = self.policy.get_v(xt_ND, t_N)
                flow_loss = self.crieria(ut_ND, v_ND_grad)
                # when computing flow_loss, remove the noise part from v_ND_grad when the batchsize is small to avoid high variance

                loss = (flow_loss + constraintloss)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                pbar.set_postfix(
                    {'Loss': '{:5f}'.format(torch.mean(loss).item()),
                     'logp': '{:5f}'.format(torch.mean(logPi_mat_NS).item()),
                     'avg distance': '{:5f}'.format(mean_distance)})
                pbar.update(1)
                #                 print(j)/
                if configs.plot_loss == True:
                    if (j + 1) % configs.plot_loss_every == 0 or j == 0:
                        loss_record = np.append(loss_record, loss.detach().cpu().numpy())
                        avg_distance_record = np.append(avg_distance_record,
                                                   mean_distance)

                if (j + 1) % configs.RLFMsave_every == 0 or j == 0:
                    #                 print(str(j) + ' Flow Loss: {:5f}'.format(loss))
                    #                 print(str(j) + ' Inside num: {:5f}'.format(torch.sum(constraint_mask)))
                    torch.save(self.policy.state_dict(),
                               './saved_model/' + configs.RLFMstage2_name + '_' + str(j + 1) + '.pth')
                    np.savez(
                        './saved_model/' + configs.RLFMstage2_name + '_' + 'train_record.npz',
                        loss_record=loss_record, avg_distance_record=avg_distance_record)

    def get_xt0(self, stage1_model, x0_ND):
        x_prev = x0_ND
        delta_t = 1 / configs.default_generation_step * self.stage1_t
        t_tensor_N = torch.zeros(x_prev.shape[0], device=configs.device, dtype=torch.float32)
        for i in range(configs.default_generation_step):
            with torch.no_grad():
                z = stage1_model(x_prev, t_tensor_N[:, None])
            x_prev = x_prev + z * 1 / configs.default_generation_step * self.stage1_t
            t_tensor_N = t_tensor_N + delta_t

        return x_prev

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
                v_ND_grad, v_ND, log_prob_N = stage2_model.get_v(x_prev, t_tensor_N, deterministic=True)

            x_prev = x_prev + v_ND * self.RL_step_width
        return x_prev



















