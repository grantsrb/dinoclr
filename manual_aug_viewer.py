import matplotlib.pyplot as plt
import argparse
import os
import sys
import datetime
import time
import math
import json
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from torchvision import datasets, transforms
from torchvision import models as torchvision_models

import utils
import vision_transformer as vits
from vision_transformer import DINOHead

import models
from main_dino import DataAugmentationDINO

def unnormalize(X):
    """
    X: torch tensor or ndarray (..., 3, H, W)
    """
    if isinstance(X, type(np.ones((1,)))):
        means = np.asarray([0.485, 0.456, 0.406])
        stds =  np.asarray([0.229, 0.224, 0.225])
    else:
        means = torch.FloatTensor([0.485, 0.456, 0.406])
        stds = torch.FloatTensor([0.229, 0.224, 0.225])
    return X*stds[:,None,None] + means[:,None,None]

if __name__=="__main__":
    img_shape = (3,32,32)
    data_path = "~/datasets/cifar10/"

    global_crops_scales = [
        (.5, 1),
        (0.9, 1),
        (0.7, 1),
        (0.5, 1),
    ]
    local_crops_scales = [
        (0.25, 0.6),
        (0.5, 0.9),
        (0.8, 0.8),
        (0.7, 0.7),
        (0.6, 0.6),
        (0.5, 0.5),
        (0.4, 0.4),
        (0.3, 0.3),
        (0.2, 0.2),
    ]
    local_crops_numbers = [
        10,
    ]
    for global_crops_scale in global_crops_scales:
        for local_crops_scale in local_crops_scales:
            for local_crops_number in local_crops_numbers:

                print("Starting new loop")
                print("global:", global_crops_scale)
                print("local crops:", local_crops_scale)
                print("local num:", local_crops_number)
                transform = DataAugmentationDINO(
                    global_crops_scale,
                    local_crops_scale,
                    local_crops_number,
                    img_shape=img_shape
                )
                dataset = datasets.CIFAR10(root=data_path, train=False, transform=transform)
                data_loader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=1,
                    drop_last=True,
                )

                for i,(imgs,y) in enumerate(data_loader):
                    for j in range(len(imgs)):
                        obs = unnormalize(imgs[j][0]).permute(1,2,0).detach().numpy()
                        plt.imshow(obs)
                        plt.show()
                    break
                break
                

