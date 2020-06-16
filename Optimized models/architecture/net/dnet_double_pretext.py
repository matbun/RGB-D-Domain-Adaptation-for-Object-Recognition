from PIL import Image
from sklearn import utils
from sklearn.model_selection import ParameterGrid
from sklearn.model_selection import train_test_split
from torch.autograd import Function
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torch.utils.data import Subset, DataLoader
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision import models
from torchvision import transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets import VisionDataset
from torchvision.models import alexnet
from torchvision.transforms.functional import pad
from tqdm import tqdm
import logging
import matplotlib.pyplot as plt
import numbers
import numpy as np
import os
import os.path
import shutil
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import zipfile

# xavier init weights
def init_weights(m):
    if type(m) == nn.Conv2d:
        nn.init.kaiming_normal_(m.weight, a=0.25)
        nn.init.zeros_(m.bias)
    elif type(m) == nn.Linear:
        m.weight.data.normal_(0, 0.01)
        m.bias.data.normal_(0, 0.01)


class DNet(nn.Module):

    def __init__(self, num_classes=1000, dim_pretext1=4, dim_pretext2=5, resnet1=None, resnet2=None):
        super(DNet, self).__init__()
        if resnet1 == None:
          resnet1 = models.resnet18(pretrained=True)
        if resnet2 == None:
          resnet2 = models.resnet18(pretrained=True)
        # The output of both "incomplete" resnet18 is 512 x 7 x 7 (C x H x W)
        # The concatenation of results is a tensor 1024 x 7 x 7

        self.model_RGB = nn.Sequential( *list(resnet1.children())[:-2])
        self.model_DEPTH = nn.Sequential( *list(resnet2.children())[:-2])
        self.Mbranch = nn.Sequential(
            # Input: 1024 x 7 x 7
            nn.AdaptiveAvgPool2d((1,1)), # -> 1024 x 1 x 1         #nn.AvgPool2d(7), #note that avgpool2d requires kernel size as parameter(in this case its input had size 7x7 so that would produce an output 1x1), while AdaptiveAvgPool2d requires output size
            nn.Flatten(),  # -> 1024 x 1
            nn.Linear(1024, 1000), # -> 1000 x 1
            nn.BatchNorm1d(1000),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(1000,num_classes)#, # -> num_classes x 1
            #nn.Softmax(dim=1)
        )
        self.Mbranch.apply(init_weights)
        self.Pbranch = nn.Sequential(
            # Input: 1024 x 7 x 7

            nn.Conv2d(1024,100, 1), # -> 100 x 7 x 7
            nn.BatchNorm2d(100),
            nn.PReLU(1),

            nn.Conv2d(100,100, 3, stride=2), # -> 100 x 5 x 5
            nn.BatchNorm2d(100),
            nn.PReLU(1),

            # Add/Remove avgpooling before fully connected
            #nn.AdaptiveAvgPool2d((1,1)), # -> 100 x 1 x 1
            nn.Flatten(), # -> 100 x 1
            nn.Linear(100*3*3,100), # -> 100 x 1
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, dim_pretext1)
            #nn.BatchNorm1d(100),
        )
        self.Pbranch.apply(init_weights)
        self.Pbranch2 = nn.Sequential(
            # Input: 1024 x 7 x 7

            nn.Conv2d(1024,100, 1), # -> 100 x 7 x 7
            nn.BatchNorm2d(100),
            nn.PReLU(1),

            nn.Conv2d(100,100, 3, stride=2), # -> 100 x 5 x 5
            nn.BatchNorm2d(100),
            nn.PReLU(1),

            # Add/Remove avgpooling before fully connected
            #nn.AdaptiveAvgPool2d((1,1)), # -> 100 x 1 x 1
            nn.Flatten(), # -> 100 x 1
            nn.Linear(100*3*3,100), # -> 100 x 1
            nn.BatchNorm1d(100),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(100, dim_pretext2)
            #nn.BatchNorm1d(100),
        )
        self.Pbranch2.apply(init_weights)

    def forward(self, x_rgb, x_depth, mode="main", debug=False):
        if mode == "main":
          x_rgb = self.model_RGB(x_rgb)
          x_depth = self.model_DEPTH(x_depth)
          x_cat = torch.cat([x_depth,x_rgb], dim=1)
          x_out = self.Mbranch(x_cat)
        elif mode == "zoom":
          x_rgb = self.model_RGB(x_rgb)
          x_depth = self.model_DEPTH(x_depth)
          x_cat = torch.cat([x_depth,x_rgb], dim=1)
          x_out = self.Pbranch2(x_cat)
        elif mode == "rotation":
            x_rgb = self.model_RGB(x_rgb)
            x_depth = self.model_DEPTH(x_depth)
            x_cat = torch.cat([x_depth,x_rgb], dim=1)
            x_out = self.Pbranch(x_cat)

        # Debug:
        if debug is True:
          print(f"RGB features shape: {x_rgb.shape}")
          print(f"Depth features shape: {x_depth.shape}")
          print(f"Concat features shape: {x_cat.shape}")
          print(f"Output shape: {x_out.shape}")

        return x_out
