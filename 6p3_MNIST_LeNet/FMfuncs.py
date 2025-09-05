import torch
import numpy as np
import tqdm
from torch.optim import Adam
from torch import nn
import torch.nn.functional as F
import configs



class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)
    def forward(self, x):
        return self.dense(x)[..., None, None]


class FMmodel(nn.Module):
    """A time-dependent score-based model built upon U-Net architecture."""

    def __init__(self, channels=[32, 64, 128, 256], embed_dim=256):
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
        self.conv1 = nn.Conv2d(2, channels[0], 3, stride=1, bias=False)
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
        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, bias=False, output_padding=1)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgnorm3 = nn.GroupNorm(32, num_channels=channels[1])
        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, bias=False, output_padding=1)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgnorm2 = nn.GroupNorm(32, num_channels=channels[0])
        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], 1, 3, stride=1)

        # The swish activation function
        self.act = lambda x: x * torch.sigmoid(x)
        # self.marginal_prob_std = marginal_prob_std

    def forward(self, x, t, x1):
        x = torch.cat((x, x1), dim=1)
        # Obtain the Gaussian random feature embedding for t
        embed = self.act(self.embed(t))
        # Encoding path
        h1 = self.conv1(x)
        ## Incorporate information from t
        h1 += self.dense1(embed)
        ## Group normalization
        h1 = self.gnorm1(h1)
        h1 = self.act(h1)
        # print(h1.shape)
        h2 = self.conv2(h1)
        h2 += self.dense2(embed)
        h2 = self.gnorm2(h2)
        h2 = self.act(h2)
        # print(h2.shape)
        h3 = self.conv3(h2)
        h3 += self.dense3(embed)
        h3 = self.gnorm3(h3)
        h3 = self.act(h3)
        # print(h3.shape)
        h4 = self.conv4(h3)
        h4 += self.dense4(embed)
        h4 = self.gnorm4(h4)
        h4 = self.act(h4)
        # print(h4.shape)

        # Decoding path
        h = self.tconv4(h4)
        ## Skip connection from the encoding path
        h += self.dense5(embed)
        h = self.tgnorm4(h)
        h = self.act(h)
        # print(h.shape)
        # print(h3.shape)
        h = self.tconv3(torch.cat([h, h3], dim=1))
        h += self.dense6(embed)
        h = self.tgnorm3(h)
        h = self.act(h)
        h = self.tconv2(torch.cat([h, h2], dim=1))
        h += self.dense7(embed)
        h = self.tgnorm2(h)
        h = self.act(h)
        h = self.tconv1(torch.cat([h, h1], dim=1))

        # Normalize output
        # h = h / marginal_prob_std(t)[:, None, None, None]
        return h





class OTFlowMatching:

    def __init__(self, sig_min: float = configs.default_sig_min) -> None:
        super().__init__()
        self.sig_min = sig_min
        self.crieria = nn.MSELoss()


    def get_untrained_model(self):

        return FMmodel().to(configs.device)


    def sample_xt_given_x1_x0(self, x0_N1TD: torch.Tensor, x1_N1tD: torch.Tensor, t_N: torch.Tensor):
        # N, D = x1_ND.shape
        std1 = self.sig_min
        return (1 - (1 - std1) * t_N[..., None, None, None]) * x0_N1TD + t_N[..., None, None, None] * x1_N1tD


    def ut_given_x1(self, xt_N1TD, x1_N1TD, t_N):
        std1 = self.sig_min
        diff = (1 - std1)
        num_ND = x1_N1TD - diff * xt_N1TD
        denom_N = 1 - diff * t_N
        return num_ND / denom_N[..., None, None, None]

    def get_samples(self, dataset, n_samples):
        dataset_size = dataset.shape[0]
        selected_ind = np.random.randint(0, dataset_size - 1, n_samples)
        return dataset[selected_ind]

    def train(self, dataset, epoches = configs.default_epoches, batch_size_N = configs.default_batchsize, lr = configs.default_lr):
        mymodel = self.get_untrained_model()
        optimizer = Adam(mymodel.parameters(), lr=lr)
        for j in tqdm.tqdm(range(epoches)):
            x1_ND = self.get_samples(dataset, batch_size_N)
            x0_ND = torch.randn_like(x1_ND, device=configs.device, dtype=torch.float32)

            t_N = torch.rand(batch_size_N, dtype=torch.float32, device=configs.device)
            xt_ND = self.sample_xt_given_x1_x0(x0_ND, x1_ND, t_N)
            ut_ND = self.ut_given_x1(xt_ND, x1_ND, t_N)
            # model_input = torch.cat([xt_ND, t_N[:, None]], dim=-1)
            vt_ND = mymodel(xt_ND, t_N[:, None], x1_ND)

            flow_loss = self.crieria(ut_ND, vt_ND)

            optimizer.zero_grad()
            flow_loss.backward()
            optimizer.step()

            if (j + 1) % configs.FMsave_every == 0 or j == 0:
                print(str(j) + ' Flow Loss: {:5f}'.format(flow_loss))
                torch.save(mymodel.state_dict(), './saved_model/'+ configs.FMmodel_name+'_' + str(j + 1) + '.pth')

        return mymodel


def sampler(FMmodel, ref, stoptime = 1, default_generation_step = configs.default_generation_step):
    x_prev = torch.randn(ref.shape[0], 1, 32, 32, dtype=torch.float32, device=configs.device)
    for i in range(default_generation_step):
        t = i/default_generation_step * stoptime
        t_tensor_N = t*torch.ones(x_prev.shape[0], device=configs.device, dtype=torch.float32)
        # input_ND = torch.cat((x_prev, t_tensor_N[:,None]), dim=1)
        with torch.no_grad():
            z = FMmodel(x_prev, t_tensor_N[:,None], ref)
        x_prev = x_prev + z * stoptime/default_generation_step
    return x_prev

#%%
# myFMmodel = FMmodel(4).to(configs.device)
# testinputx = torch.randn(10,4).to(configs.device)
# testinputt = torch.randn(10,1).to(configs.device)
#
# myFMmodel(testinputx,testinputt)