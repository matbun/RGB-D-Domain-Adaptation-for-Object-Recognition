from torchvision.datasets import ImageFolder
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torch.autograd import Function
import logging
import numpy as np
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
import torchvision
from torchvision import transforms
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import VisionDataset
import os.path
import sys
from torch.utils.data import DataLoader
from torchvision.transforms.functional import pad
import numbers
import torch.nn.functional as F
from torchvision import models
import types


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')

def zoom_img(img, zoom_perc):
  """
  img: PIL image
  zoom_perc: [0, 100]
  Return cropped_img
  """
  zoom_perc /= 100
  # images given to this method are already transformed to square images by the pre zoom transform
  # to avoid inconsistencies, the min(width, height) is taken as resize shape. THis is the value that is used to
  # decide the size of the zoom reduction
  width, height = img.size
  resize_shape = min(width, height)

  crop_size = int(resize_shape * (1-zoom_perc))

  transformer = transforms.Compose([
                                    transforms.CenterCrop(crop_size)
                                  ])
  return transformer(img)

def uniform_difference_couple(max_val):
  """
  Returns 0 and an integer uniformly asmpled in the interval [0, max_val].
  0 is randomly assigned to int1 or int2.
  Return int1, int2, diff
  """
  diff = np.random.randint(0, max_val + 1)
  b = np.random.binomial(n=1, p=0.5)
  if b:
    return 0, diff, diff
  return diff, 0, diff

def zoom_task_extractor(rgb_image, depth_image, label, parameters_dict):
        MAX_ZOOM_PERCENTAGE = parameters_dict["MAX_ZOOM_PERCENTAGE"]
        PRE_ZOOM_TRANSFORM = parameters_dict["PRE_ZOOM_TRANSFORM"] # Preprocessing before zoom. Same for RGB and D
        MAIN_RGB_TRANSFORM = parameters_dict["MAIN_RGB_TRANSFORM"] # Preprocessing for original RGB image
        MAIN_DEPTH_TRANSFORM = parameters_dict["MAIN_DEPTH_TRANSFORM"] # Preprocessing for original D image
        POST_ZOOM_RGB_TRANSFORM = parameters_dict["POST_ZOOM_RGB_TRANSFORM"] # Preprocessing for zoomed RGB image
        POST_ZOOM_DEPTH_TRANSFORM = parameters_dict["POST_ZOOM_DEPTH_TRANSFORM"] # Preprocessing for zoomed D image

        zoom_perc1, zoom_perc2, pretext_task_label = uniform_difference_couple(MAX_ZOOM_PERCENTAGE)

        preprocessed_rgb_image = PRE_ZOOM_TRANSFORM(rgb_image)
        preprocessed_depth_image = PRE_ZOOM_TRANSFORM(depth_image)

        if zoom_perc1 > 0:
            zoomed_rgb_image = zoom_img(preprocessed_rgb_image, zoom_perc1)
        else:
            zoomed_rgb_image = preprocessed_rgb_image
        
        if zoom_perc2 > 0:
            zoomed_depth_image = zoom_img(preprocessed_depth_image, zoom_perc2)
        else:
            zoomed_depth_image = preprocessed_depth_image

        if MAIN_RGB_TRANSFORM is not None and MAIN_DEPTH_TRANSFORM is not None:
            rgb_image = MAIN_RGB_TRANSFORM(rgb_image)
            depth_image = MAIN_DEPTH_TRANSFORM(depth_image)
        if POST_ZOOM_DEPTH_TRANSFORM is not None and POST_ZOOM_RGB_TRANSFORM is not None:
            zoomed_rgb_image = POST_ZOOM_RGB_TRANSFORM(zoomed_rgb_image)
            zoomed_depth_image = POST_ZOOM_DEPTH_TRANSFORM(zoomed_depth_image)
        return (rgb_image, depth_image, label), (zoomed_rgb_image, zoomed_depth_image, pretext_task_label)

def decentralized_zoom_task_extractor(rgb_image, depth_image, label, parameters_dict):
        MAX_ZOOM_PERCENTAGE = parameters_dict["MAX_ZOOM_PERCENTAGE"]
        PRE_ZOOM_TRANSFORM = parameters_dict["PRE_ZOOM_TRANSFORM"] # Preprocessing before zoom. Same for RGB and D
        MAIN_RGB_TRANSFORM = parameters_dict["MAIN_RGB_TRANSFORM"] # Preprocessing for original RGB image
        MAIN_DEPTH_TRANSFORM = parameters_dict["MAIN_DEPTH_TRANSFORM"] # Preprocessing for original D image
        POST_ZOOM_RGB_TRANSFORM = parameters_dict["POST_ZOOM_RGB_TRANSFORM"] # Preprocessing for zoomed RGB image
        POST_ZOOM_DEPTH_TRANSFORM = parameters_dict["POST_ZOOM_DEPTH_TRANSFORM"] # Preprocessing for zoomed D image
    
        # Preprocessing
        preprocessed_rgb_image = PRE_ZOOM_TRANSFORM(rgb_image)
        preprocessed_depth_image = PRE_ZOOM_TRANSFORM(depth_image)
        
        pretext_task_label = np.random.randint(0, MAX_ZOOM_PERCENTAGE + 1)
        choose_crop = np.random.randint(0, 5) # top-left, top-right, center...
        b = np.random.binomial(n=1, p=0.5) # who crop? toss coin...
        
        crop_size = int(224 * (1-pretext_task_label/100))
        five_crop = transforms.FiveCrop(crop_size)
        if b:
            # zoomed_rgb_image
            top_left, top_right, bottom_left, bottom_right, center = five_crop(preprocessed_rgb_image)
            crops = [top_left, top_right, bottom_left, bottom_right, center]
            zoomed_rgb_image = crops[choose_crop]
            # zoomed_depth_image
            zoomed_depth_image = preprocessed_depth_image
        else:
            # zoomed_rgb_image
            zoomed_rgb_image = preprocessed_rgb_image
            # zoomed_depth_image
            top_left, top_right, bottom_left, bottom_right, center = five_crop(preprocessed_depth_image)
            crops = [top_left, top_right, bottom_left, bottom_right, center]
            zoomed_depth_image = crops[choose_crop]
        
        # Postprocessing
        if MAIN_RGB_TRANSFORM is not None and MAIN_DEPTH_TRANSFORM is not None:
            rgb_image = MAIN_RGB_TRANSFORM(rgb_image)
            depth_image = MAIN_DEPTH_TRANSFORM(depth_image)
        if POST_ZOOM_DEPTH_TRANSFORM is not None and POST_ZOOM_RGB_TRANSFORM is not None:
            zoomed_rgb_image = POST_ZOOM_RGB_TRANSFORM(zoomed_rgb_image)
            zoomed_depth_image = POST_ZOOM_DEPTH_TRANSFORM(zoomed_depth_image)
        return (rgb_image, depth_image, label), (zoomed_rgb_image, zoomed_depth_image, pretext_task_label)
        
        
def relative_rot_task_extractor(rgb_image, depth_image, label, parameters_dict):
        MAIN_RGB_TRANSFORM = parameters_dict["MAIN_RGB_TRANSFORM"] # Preprocessing for original RGB image
        MAIN_DEPTH_TRANSFORM = parameters_dict["MAIN_DEPTH_TRANSFORM"] # Preprocessing for original RGB image
        PRETEXT_RGB_TRANSFORM = parameters_dict["PRETEXT_RGB_TRANSFORM"] # Preprocessing for rotated RGB image
        PRETEXT_DEPTH_TRANSFORM = parameters_dict["PRETEXT_DEPTH_TRANSFORM"] # Preprocessing for rotated D image

        j, k = np.random.randint(0, 4, 2)
        #j is the rotation index for rgb image
        #k is the rotation index for depth image
        z_encoding = (k-j) % 4
        # j       k
        rgb_rot, depth_rot = j, k

        rotated_rgb_image = rgb_image.rotate(rgb_rot*(-90))
        rotated_depth_image = depth_image.rotate(depth_rot*(-90))

        if MAIN_RGB_TRANSFORM is not None and MAIN_DEPTH_TRANSFORM is not None :
            rgb_image = MAIN_RGB_TRANSFORM(rgb_image)
            depth_image = MAIN_DEPTH_TRANSFORM(depth_image)
        if PRETEXT_RGB_TRANSFORM is not None and PRETEXT_DEPTH_TRANSFORM is not None:
            rotated_rgb_image = PRETEXT_RGB_TRANSFORM(rotated_rgb_image)
            rotated_depth_image = PRETEXT_DEPTH_TRANSFORM(rotated_depth_image)
        return (rgb_image, depth_image, label), (rotated_rgb_image, rotated_depth_image, z_encoding)
