import torch
import torchvision
import torchvision.transforms as transforms
import cv2
import numpy as np
import matplotlib.pyplot as plt


def thickness_once(image_1NN): 
    image_np = image_1NN.squeeze().numpy()  # Convert from (1, 28, 28) to (28, 28)
    image_np = np.clip(image_np, 0, 1)
    # Convert to 8-bit grayscale
    image_np = (image_np * 255).astype(np.uint8)

    # Invert colors (MNIST is black-on-white, but distance transform expects white-on-black)
    binary_image = cv2.threshold(image_np, 128, 255, cv2.THRESH_BINARY)[1]

    # Compute the distance transform
    dist_transform = cv2.distanceTransform(binary_image, distanceType=cv2.DIST_L2, maskSize=5)

    # Thickness metric: Maximum value in the distance transform
    thickness = np.max(dist_transform)
    return thickness

def thickness_batch(image_batch):
    thickness_vec = np.zeros(image_batch.shape[0])
    for i in range(image_batch.shape[0]):
        thickness_vec[i] = thickness_once(image_batch[i])
    return thickness_vec

def brightness_batch(image_batch_in):
    image_batch = image_batch_in.clone()
    image_batch[image_batch >0.5] = 1
    image_batch[image_batch <=0.5] = 0
    return torch.sum(image_batch, dim=(1,2,3))