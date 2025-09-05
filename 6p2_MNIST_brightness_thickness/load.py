import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt
from compute_thickness import thickness_batch, brightness_batch
import configs

class myMMIST:
    def __init__(self):
        transform = transforms.Compose([
            transforms.ToTensor(),  # Convert PIL image to tensor
        ])

        mnist_dataset_train = torchvision.datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        self.dataloader_train = torch.utils.data.DataLoader(mnist_dataset_train, batch_size=70000, shuffle=True)

        mnist_dataset_test = torchvision.datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        self.dataloader_test = torch.utils.data.DataLoader(mnist_dataset_test, batch_size=70000, shuffle=True)

    def get_data(self, train=True, min_thickness = configs.min_thickness, max_thickness = configs.max_thickness, plot_hist = False):
        if train:
            tmp_data_loader  = self.dataloader_train
            print("Getting the valid data in the training dataset")
        else:
            tmp_data_loader = self.dataloader_test
            print("Getting the valid data in the training dataset")
        
        print("Minimum Thickness: ", min_thickness)
        print("Maximum Thickness: ", max_thickness)
        for image_batch, label_batch in tmp_data_loader:
            thickness_vec = thickness_batch(image_batch)
            valid_ind = np.where((thickness_vec > min_thickness) & (thickness_vec < max_thickness))[0]
            valid_set = image_batch[valid_ind]

        print("The total number of original data is ", len(image_batch))
        print("The total number of valid data is ", len(valid_set))

        if plot_hist:
            plt.figure(figsize=(8, 5))
            plt.hist(thickness_vec, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel("Thickness Value")
            plt.ylabel("Frequency")
            plt.title("Histogram of Thickness Values")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.show()

            plt.figure(figsize=(8, 5))
            plt.hist(label_batch[valid_ind], bins=10, edgecolor='black', alpha=0.7)
            plt.xlabel("Number")
            plt.ylabel("Frequency")
            plt.title("Number Frequency of valid data")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Show the plot
            plt.show()
        return valid_set

    def get_data_brightness(self, train=True, min_brightness=configs.min_brightness, max_brightness=configs.max_brightness,
                 plot_hist=False):
        if train:
            tmp_data_loader = self.dataloader_train
            print("Getting the valid data in the training dataset")
        else:
            tmp_data_loader = self.dataloader_test
            print("Getting the valid data in the training dataset")

        print("Minimum brightness: ", min_brightness)
        print("Maximum brightness: ", max_brightness)
        for image_batch, label_batch in tmp_data_loader:
            brightness_vec = brightness_batch(image_batch)
            valid_ind = np.where((brightness_vec > min_brightness) & (brightness_vec < max_brightness))[0]
            valid_set = image_batch[valid_ind]

        print("The total number of original data is ", len(image_batch))
        print("The total number of valid data is ", len(valid_set))

        if plot_hist:
            plt.figure(figsize=(8, 5))
            plt.hist(brightness_vec, bins=50, edgecolor='black', alpha=0.7)
            plt.xlabel("brightness Value")
            plt.ylabel("Frequency")
            plt.title("Histogram of brightness Values")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            plt.show()

            plt.figure(figsize=(8, 5))
            plt.hist(label_batch[valid_ind], bins=10, edgecolor='black', alpha=0.7)
            plt.xlabel("Number")
            plt.ylabel("Frequency")
            plt.title("Number Frequency of valid data")
            plt.grid(axis='y', linestyle='--', alpha=0.7)

            # Show the plot
            plt.show()
        return valid_set

