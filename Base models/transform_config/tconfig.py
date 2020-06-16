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

class TransformConfig:
  def __init__(self, resize_shape=256, centercrop_shape=224): #, config_type="imagenet", pretext="rotation"
    self.init_means = (0.5, 0.5, 0.5)
    self.init_std = (0.5, 0.5, 0.5)
    self.imgnet_means = (0.485, 0.456, 0.406)
    self.imgnet_std = (0.229, 0.224, 0.225)

    # NORMALIZATION BUILDING BLOCKS
    self.standard_transform = transforms.Compose([transforms.Resize(resize_shape),
                                      transforms.CenterCrop(centercrop_shape),
                                      transforms.ToTensor(),
                                      transforms.Normalize(self.imgnet_means, self.imgnet_std)])
    self.normalize_synrod_rgb = transforms.Normalize((0.3094721, 0.2854524, 0.24852814), \
                                                (0.2328845, 0.22429037, 0.22981688))
    self.normalize_synrod_depth = transforms.Normalize((0.44173935, 0.4029698, 0.6806104), \
                                                  (0.13479522, 0.2542602, 0.36353043))

    self.normalize_rod_rgb = transforms.Normalize((0.5488044, 0.52112335, 0.5065665), \
                                            (0.21050504, 0.22136924, 0.25390014))
    self.normalize_rod_depth = transforms.Normalize((0.7427522, 0.29610398, 0.47648674), \
                                              (0.20184048, 0.20392625, 0.2904062))


    # invariant for imagenet or depth or rgb
    self.pre_zoom_transform = transforms.Compose([                      #select 256 to have a slight zoom when centercropping to 224  # Zoom to fit the image in 224 x 224
                                              transforms.Resize(resize_shape),  # or select 224 to not have an initial slight zoom on the image
                                              transforms.CenterCrop(centercrop_shape) # -> 224 x 224 as ResNets expected input
                                              ])


    # MAIN TASK TRANSFORMATIONS --> VALID ALSO FOR ROTATION TASK(PIPELINE FOR TRANSFORMING ROTATED IMAGES IS EQUAL TO THAT OF THE MAIN TASK)
    self.synrod_rgb_transform = transforms.Compose([transforms.Resize(resize_shape),
                                          transforms.CenterCrop(centercrop_shape),
                                          transforms.ToTensor(),
                                          self.normalize_synrod_rgb])
    self.synrod_depth_transform = transforms.Compose([transforms.Resize(resize_shape),
                                          transforms.CenterCrop(centercrop_shape),
                                          transforms.ToTensor(),
                                          self.normalize_synrod_depth])
    self.rod_rgb_transform = transforms.Compose([transforms.Resize(resize_shape),
                                          transforms.CenterCrop(centercrop_shape),
                                          transforms.ToTensor(),
                                          self.normalize_rod_rgb])
    self.rod_depth_transform = transforms.Compose([transforms.Resize(resize_shape),
                                          transforms.CenterCrop(centercrop_shape),
                                          transforms.ToTensor(),
                                          self.normalize_rod_depth])


    # POST ZOOM TRANSFORMATIONS
    self.standard_post_zoom_transform = transforms.Compose([
                                            transforms.Resize(resize_shape),
                                            transforms.ToTensor(),
                                            transforms.Normalize(self.imgnet_means, self.imgnet_std)])

    self.synrod_post_zoom_rgb_transform = transforms.Compose([
                                            transforms.Resize(resize_shape),
                                            transforms.ToTensor(),
                                            self.normalize_synrod_rgb])
    self.synrod_post_zoom_depth_transform = transforms.Compose([
                                            transforms.Resize(resize_shape),
                                            transforms.ToTensor(),
                                            self.normalize_synrod_depth])

    self.rod_post_zoom_rgb_transform = transforms.Compose([
                                            transforms.Resize(resize_shape),
                                            transforms.ToTensor(),
                                            self.normalize_rod_rgb])

    self.rod_post_zoom_depth_transform = transforms.Compose([
                                            transforms.Resize(resize_shape),
                                            transforms.ToTensor(),
                                            self.normalize_rod_depth])

  def get_zoom_configuration(self, config_type="imagenet", max_percent_zoom_value=40.0):
    if config_type == "imagenet":
      synrod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.standard_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.standard_post_zoom_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.standard_post_zoom_transform}

      rod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.standard_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.standard_post_zoom_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.standard_post_zoom_transform}
    elif config_type == "rgb_mod":
      synrod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.synrod_rgb_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.synrod_post_zoom_rgb_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.standard_post_zoom_transform}

      rod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.rod_rgb_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.rod_post_zoom_rgb_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.standard_post_zoom_transform}

    elif config_type == "depth_mod":
      synrod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.standard_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.synrod_depth_transform ,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.standard_post_zoom_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.synrod_post_zoom_depth_transform}

      rod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.standard_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.rod_depth_transform,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.standard_post_zoom_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.rod_post_zoom_depth_transform}
    elif config_type == "rgb_depth_mod":
      synrod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.synrod_rgb_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.synrod_depth_transform ,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.synrod_post_zoom_rgb_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.synrod_post_zoom_depth_transform}

      rod_zoom_task_param_values = {"MAX_ZOOM_PERCENTAGE": max_percent_zoom_value,
                                        "MAIN_RGB_TRANSFORM" : self.rod_rgb_transform,
                                        "MAIN_DEPTH_TRANSFORM": self.rod_depth_transform,
                                        "PRE_ZOOM_TRANSFORM": self.pre_zoom_transform,
                                        "POST_ZOOM_RGB_TRANSFORM":self.rod_post_zoom_rgb_transform,
                                        "POST_ZOOM_DEPTH_TRANSFORM":self.rod_post_zoom_depth_transform}
    return synrod_zoom_task_param_values, rod_zoom_task_param_values

  def get_rotation_configuration(self, config_type="imagenet"):
    if config_type == "imagenet":
      synrod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.standard_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.standard_transform}
      rod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.standard_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.standard_transform}
    elif config_type == "rgb_mod":
      synrod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.synrod_rgb_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.synrod_rgb_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.standard_transform}
      rod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.rod_rgb_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.rod_rgb_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.standard_transform}

    elif config_type == "depth_mod":
      synrod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.standard_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.synrod_depth_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.synrod_depth_transform}
      rod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.standard_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.rod_depth_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.standard_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.rod_depth_transform}
    elif config_type == "rgb_depth_mod":
      synrod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.synrod_rgb_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.synrod_depth_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.synrod_rgb_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.synrod_depth_transform}
      rod_rotation_task_param_values = {"MAIN_RGB_TRANSFORM" : self.rod_rgb_transform,
                                                      "MAIN_DEPTH_TRANSFORM": self.rod_depth_transform,
                                                      "PRETEXT_RGB_TRANSFORM": self.rod_rgb_transform,
                                                      "PRETEXT_DEPTH_TRANSFORM": self.rod_depth_transform}
    return synrod_rotation_task_param_values, rod_rotation_task_param_values
