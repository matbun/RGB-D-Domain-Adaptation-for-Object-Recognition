from torchvision.datasets import ImageFolder
import os
import shutil
import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url as load_state_dict_from_url
from torchvision.models import alexnet
import torch.nn as nn
from torch.autograd import Function
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Subset, DataLoader
from torch.backends import cudnn
import torchvision
from torchvision import transforms
from torchvision.models import alexnet
from PIL import Image
from tqdm import tqdm
from torchvision.datasets import VisionDataset
from PIL import Image
import os
import os.path
import sys
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn import utils
from torchvision.transforms.functional import pad
from torchvision import transforms
import numpy as np
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import matplotlib.pyplot as plt
import types

class HLoss(nn.Module):
    def __init__(self):
        super(HLoss, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=1) * F.log_softmax(x, dim=1)
        b = -1.0 * b.sum()
        return b
# DATALOADER function
def collate(batch):
  return batch

# helper function used to setup batches returned by the dataloaders in the way that is mentioned in the paper
def format_batch(batch, pretext_task="rotation"):
  """"
  set pretext_task == rotation to require the pretext task labels to be of type "long"
  set pretext_task == zoomreg to require the pretext task labels to be of type "float"
  set pretext_task == zoomclf to require the pretext task labels to be of type "float"
  """
  data = {"rgb":[], "depth":[], "label":[] }
  data_hat = {"rgb":[], "depth":[], "label":[] }
  for tuple_, tuple_hat in batch:
    rgb_img, depth_img, label = tuple_
    rot_rgb_img, rot_depth_img, rot_label = tuple_hat

    data["rgb"].append(rgb_img[None,:])
    data["depth"].append(depth_img[None,:])
    data["label"].append(label)

    data_hat["rgb"].append(rot_rgb_img[None,:])
    data_hat["depth"].append(rot_depth_img[None,:])
    data_hat["label"].append(rot_label)

  data["rgb"] = torch.cat(data["rgb"] , dim=0)
  data["depth"] = torch.cat(data["depth"] , dim=0)
  data["label"] = torch.LongTensor(data["label"])

  data_hat["rgb"] = torch.cat(data_hat["rgb"] , dim=0)
  data_hat["depth"] = torch.cat(data_hat["depth"] , dim=0)
  if pretext_task == "rotation" or pretext_task == "zoomclf":
    data_hat["label"] = torch.LongTensor(data_hat["label"] )
  elif pretext_task == "zoomreg":
    data_hat["label"] = torch.FloatTensor(data_hat["label"] )

  return data, data_hat


#side by side loss and accuracy plot
def make_plot(train_loss, train_acc, test_loss, test_acc):
  f = plt.figure(figsize=(10,3))
  ax1 = f.add_subplot(121)
  ax2 = f.add_subplot(122)

  # plot all points registered during training
  ax1.plot(range(0,len(train_loss)), train_loss, label="train")
  ax1.plot(range(0, len(test_loss)), test_loss, label="test")

  # or average them for each epoch and plot per epoch
  #ax1.plot(range(0, num_epochs), train_loss, label="train")
  #ax1.plot(range(0, num_epochs), test_loss, label="test")
  ax1.set_title("loss")
  ax1.grid()
  #uncomment this to decide scale of the loss plot
  #ax1.set_ylim(0, 5)
  ax1.legend()
  ax2.plot(range(0, len(train_acc)),train_acc , label="train")
  ax2.plot(range(0, len(test_acc)), test_acc, label="test")

  #ax2.plot(range(0, num_epochs),train_acc , label="train")
  #ax2.plot(range(0, num_epochs), test_acc, label="test")
  ax2.set_title("accuracy")
  #uncomment this to decide scale of the accuracy plot
  #ax2.set_ylim(0,1.05)
  ax2.grid()
  ax2.legend()

def learning_curves(training_accuracies, training_losses, validation_accuracies, validation_losses, plot_title, plot_size=(16,6)):
  """
  Plots accuracies and losses per epochs.
  """
  fig, ax = plt.subplots(nrows=1, ncols=2, figsize=plot_size)
  ax[0].plot(range(1,len(training_accuracies)+1), training_accuracies, label="Training")
  ax[0].plot(range(1,len(validation_accuracies)+1), validation_accuracies, label="Validation")
  ax[0].legend()
  ax[0].set_title("Accuracy")
  ax[0].set_xlabel("Epochs")

  ax[1].plot(range(1,len(training_losses)+1), training_losses, label="Training")
  ax[1].plot(range(1,len(validation_losses)+1), validation_losses, label="Validation")
  ax[1].legend()
  ax[1].set_title("Loss")
  ax[1].set_xlabel("Epochs")

  fig.suptitle(plot_title)
  plt.show()

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

def clf_uniform_difference_couple(max_val):
  """
  Returns 0 and an integer uniformly asmpled in the interval [0, max_val].
  0 is randomly assigned to int1 or int2.
  Return int1, int2, diff
  """
  #diff = np.random.randint(0, max_val + 1)
  #diff = np.random.uniform(0, max_val)
  diff = np.random.randint(0, 5) # for classification

  b = np.random.binomial(n=1, p=0.5)
  if b:
    return 0, diff*10, diff
  return diff*10, 0, diff

# for regression
def reg_uniform_difference_couple(max_val):
  """
  Returns 0 and an integer uniformly asmpled in the interval [0, max_val].
  0 is randomly assigned to int1 or int2.
  Return int1, int2, diff
  """
  #diff = np.random.randint(0, max_val + 1)
  diff = np.random.uniform(0, max_val)
  #diff = np.random.randint(0, 6) # for classification

  b = np.random.binomial(n=1, p=0.5)
  if b:
    return 0, diff, diff
  return diff, 0, diff

def zoom_task_extractor(rgb_image, depth_image, label, parameters_dict, task_type="clf"):
        MAX_ZOOM_PERCENTAGE = parameters_dict["MAX_ZOOM_PERCENTAGE"]
        PRE_ZOOM_TRANSFORM = parameters_dict["PRE_ZOOM_TRANSFORM"] # Preprocessing before zoom. Same for RGB and D
        MAIN_RGB_TRANSFORM = parameters_dict["MAIN_RGB_TRANSFORM"] # Preprocessing for original RGB image
        MAIN_DEPTH_TRANSFORM = parameters_dict["MAIN_DEPTH_TRANSFORM"] # Preprocessing for original D image
        POST_ZOOM_RGB_TRANSFORM = parameters_dict["POST_ZOOM_RGB_TRANSFORM"] # Preprocessing for zoomed RGB image
        POST_ZOOM_DEPTH_TRANSFORM = parameters_dict["POST_ZOOM_DEPTH_TRANSFORM"] # Preprocessing for zoomed D image

        if task_type == "clf":
            zoom_perc1, zoom_perc2, pretext_task_label = clf_uniform_difference_couple(MAX_ZOOM_PERCENTAGE)
        elif task_type == "reg":
            zoom_perc1, zoom_perc2, pretext_task_label = reg_uniform_difference_couple(MAX_ZOOM_PERCENTAGE)
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
