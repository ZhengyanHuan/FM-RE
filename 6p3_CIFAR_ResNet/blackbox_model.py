from externel.resnet_models import *
import torch
import configs
import os

class black_box_model_class:
    def __init__(self, device = configs.device):
        trained_models_directory = "externel/"
        model_file_name = "resnet50_cifar10_lr01.pth"
        device_name = 'cuda:0'

        path = os.path.join(trained_models_directory, model_file_name)

        net = ResNet50()
        net.load_state_dict(torch.load(path, map_location=device_name)['net'])

        ## model in evaluation mode
        net.eval()
        self.model = net.to(device)
        self.celoss = torch.nn.CrossEntropyLoss(reduction='none')
        print("Blackbox model is loadded!")
    def predict(self, x_B3nn):
        outputs = self.model(x_B3nn)
        _, predicted = torch.max(outputs.data, 1)
        return predicted

    def predict_proba(self, x_B3nn):
        with torch.no_grad():
            outputs = self.model(x_B3nn)
        return outputs

    def CEloss(self, x_B3nn, y_B):
        with torch.no_grad():
            outputs = self.model(x_B3nn)
        return self.celoss(outputs, y_B)