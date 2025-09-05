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


class FMmodel(nn.Module):
    def __init__(self, d_model,t_layer_num = 2, x_layer_num = 3, tx_layer_num = 3, hidden_dim = 64):
        super(FMmodel, self).__init__()
        self.t_net = get_linear_layer_block(t_layer_num, 1, hidden_dim, hidden_dim)
        self.x_net = get_linear_layer_block(x_layer_num, d_model, hidden_dim, hidden_dim)
        self.tx_net = get_linear_layer_block(tx_layer_num, hidden_dim, hidden_dim, d_model)

    def forward(self, x, t):
        combined_tx = self.t_net(t) + self.x_net(x)
        combined_tx = F.selu(combined_tx)
        out = self.tx_net(combined_tx)
        return out


class OTFlowMatching:

    def __init__(self, sig_min: float = configs.default_sig_min, d_model = configs.d_model) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.crieria = nn.MSELoss()
        self.d_model = d_model

    def get_untrained_model(self):

        return FMmodel(self.d_model).to(configs.device)


    def sample_xt_given_x1_x0(self, x0_ND: torch.Tensor, x1_ND: torch.Tensor, t_N: torch.Tensor):
        # N, D = x1_ND.shape
        std1 = self.sig_min
        return (1 - (1 - std1) * t_N[..., None]) * x0_ND + t_N[..., None] * x1_ND


    def ut_given_x1(self, xt_ND, x1_ND, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_ND - diff * xt_ND
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None]

    def get_samples(self, dataset, n_samples):
        dataset_size = dataset.shape[0]
        selected_ind = np.random.randint(0, dataset_size - 1, n_samples)
        return dataset[selected_ind]

    def train(self, dataset, epoches = configs.default_epoches, batch_size_N = configs.default_batchsize, lr = configs.default_lr, reflect = False):  #reflect is for implementing RFM, a baseline
        mymodel = self.get_untrained_model()
        optimizer = Adam(mymodel.parameters(), lr=lr)
        for j in tqdm.tqdm(range(epoches)):
            x1_ND = self.get_samples(dataset, batch_size_N)
            x0_ND = torch.randn_like(x1_ND, device=configs.device, dtype=torch.float32)
            if reflect:
                x0_ND = self.dual_mirror_proj(x0_ND) # ensure the starting distribution in the ball
                # print(x1_ND)


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
                torch.save(mymodel.state_dict(), './saved_model/'+ configs.FMmodel_name+'_' + str(j + 1) + '.pth')

        return mymodel

    def mirror_proj(self, x1_ND, gamma = 1, R=1.000):
        num = 2*gamma*x1_ND
        denom = R - torch.norm(x1_ND, dim=1)**2
        return num / denom[:,None]

    def dual_mirror_proj(self, x1_ND, gamma = 1, R=1):
        num = R * x1_ND

        denom = gamma + torch.sqrt(R * torch.norm(x1_ND, dim=1) ** 2 + gamma ** 2)

        return num / denom[:,None]

def check_outside_l2ball(mat_Ndim):
    return torch.norm(mat_Ndim, dim=-1)>1

def reflect4ball(x_prev):
    flags = check_outside_l2ball(x_prev)
    while torch.sum(flags) >0:
        x_prev[flags] = (2/torch.norm(x_prev[flags], dim=1)-1)[:,None]*x_prev[flags]
        flags = check_outside_l2ball(x_prev)

    return x_prev

def dual_mirror_proj_out( x1_ND, gamma = 1, R=1):
    num = R * x1_ND

    denom = gamma + torch.sqrt(R * torch.norm(x1_ND, dim=1) ** 2 + gamma ** 2)

    return num / denom[:,None]

def sampler(FMmodel, batch_size, stoptime = 1, reflect = False): #reflect is for implementing RFM, a baseline
    x_prev = torch.randn(batch_size, configs.d_model, dtype=torch.float32, device=configs.device)
    # if reflect:
        # x_prev = dual_mirror_proj_out(x_prev)
    for i in range(configs.default_generation_step):
        t = i/configs.default_generation_step * stoptime
        t_tensor_N = t*torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32)
        # input_ND = torch.cat((x_prev, t_tensor_N[:,None]), dim=1)
        with torch.no_grad():
            z = FMmodel(x_prev, t_tensor_N[:,None])
        x_prev = x_prev + z * 1/configs.default_generation_step
        # if reflect:
            # x_prev = reflect4ball(x_prev)
    return x_prev

#%%
# myFMmodel = FMmodel(4).to(configs.device)
# testinputx = torch.randn(10,4).to(configs.device)
# testinputt = torch.randn(10,1).to(configs.device)
#
# myFMmodel(testinputx,testinputt)