import torch
import torch.nn as nn
import numpy as np
from scipy import linalg
from collections import OrderedDict

from torch.cuda import device
import torchvision.transforms as transforms
import torch.nn.functional as F

class LeNet5(nn.Module):

    def __init__(self):
        super(LeNet5, self).__init__()

        self.convnet = nn.Sequential(OrderedDict([
            ('c1', nn.Conv2d(1, 6, kernel_size=(5, 5))),
            ('relu1', nn.ReLU()),
            ('s2', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c3', nn.Conv2d(6, 16, kernel_size=(5, 5))),
            ('relu3', nn.ReLU()),
            ('s4', nn.MaxPool2d(kernel_size=(2, 2), stride=2)),
            ('c5', nn.Conv2d(16, 120, kernel_size=(5, 5))),
            ('relu5', nn.ReLU())
        ]))

        self.fc = nn.Sequential(OrderedDict([
            ('f6', nn.Linear(120, 84)),
            ('relu6', nn.ReLU()),
            ('f7', nn.Linear(84, 10)),
            ('sig7', nn.LogSoftmax(dim=-1))
        ]))

    def forward(self, img):
        output = self.convnet(img)
        output = output.view(img.size(0), -1)
        output = self.fc(output)
        return output

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = LeNet5().eval()
net.load_state_dict(torch.load('./fid/lenet_epoch=12_test_acc=0.991.pth', weights_only=True))
net = net.to(device)
resize_to_32 = transforms.Resize((32, 32))

def extract_f6_features_batch( images, device = device, model = net):

    model.eval()
    images = images.to(device)

    with torch.no_grad():
        x = model.convnet(images).view(images.size(0), -1)
        f6_output = model.fc[0](x)       # Linear(120 → 84)
        f6_output = model.fc[1](f6_output)  # ReLU

    return f6_output.cpu().numpy()


def calculate_fid_from_features(features1, features2, eps=1e-6):
    mu1 = np.mean(features1, axis=0)
    mu2 = np.mean(features2, axis=0)
    sigma1 = np.cov(features1, rowvar=False)
    sigma2 = np.cov(features2, rowvar=False)

    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)

    if np.iscomplexobj(covmean):
        covmean = covmean.real

    fid = np.sum((mu1 - mu2)**2) + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid


def compute_fid_from_batches(batch1, batch2, device= device, model = net):
    """
    batch1, batch2: Tensors of shape (B, 1, N, N)
    """
    features1 = extract_f6_features_batch( batch1, device, model)
    features2 = extract_f6_features_batch(batch2, device, model)
    return calculate_fid_from_features(features1, features2)
