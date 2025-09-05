from externel.lenet import LeNet5
import torch
import configs

class black_box_model_class:
    def __init__(self, device = configs.device):

        net = LeNet5().eval()
        net.load_state_dict(torch.load('externel/lenet_epoch=12_test_acc=0.991.pth'))
        self.model = net.to(device)

    def predict(self, x_B1nn):
        outputs = self.model(x_B1nn)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def predict_proba(self, x_B1nn):
        with torch.no_grad():
            outputs = self.model(x_B1nn)
        return outputs