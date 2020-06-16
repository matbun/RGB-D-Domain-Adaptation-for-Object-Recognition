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

from rod_utils import *  # synROd and ROD useful functions


class SynRODMOD(VisionDataset):
    """
    item_extractor_fn: accepts either a string to specify one of the existing types of task ("rotation" or "zoom")
                                    and the respective function is automatically selected

                        or specify your own function to do the extraction
                        if it is a function it has to be created following this synthax

                        def func_name(rgb_image, depth_image, original_label, item_extractor_param_values)


    item_extractor_param_values: (dictionary) that specifies for each key, the value of a parameter
                                    that has to be passed to the provided item_extractor_fn

                                ROTATION TASK DICTIONARY MUST SPECIFY THE VALUES FOR THE KEYS:

                                                                                "MAIN_RGB_TRANSFORM": Preprocessing for original RGB image.
                                                                                "MAIN_DEPTH_TRANSFORM": Preprocessing for original D image.
                                                                                "PRETEXT_RGB_TRANSFORM": Preprocessing for rotated RGB image.
                                                                                "PRETEXT_DEPTH_TRANSFORM": : Preprocessing for rotated D image.

                                ZOOM TASK DICTIONARY MUST SPECIFY THE VALUES FOR THE KEYS:

                                                                                "MAX_ZOOM_PERCENTAGE"
                                                                                "PRE_ZOOM_TRANSFORM": Preprocessing before zoom. Same for RGB and D.
                                                                                "MAIN_RGB_TRANSFORM": Preprocessing for original RGB image.
                                                                                "MAIN_DEPTH_TRANSFORM": Preprocessing for original D image.
                                                                                "POST_ZOOM_RGB_TRANSFORM": Preprocessing for zoomed RGB image.
                                                                                "POST_ZOOM_DEPTH_TRANSFORM": Preprocessing for zoomed D image.



    """

    def __init__(self,
                 root,
                 item_extractor_fn=None,
                 item_extractor_param_values=None,
                 split_path=None
                 ):
        super(SynRODMOD, self).__init__(root)

        self.pretext_task_type = item_extractor_fn
        if isinstance(item_extractor_fn, types.FunctionType):
            self.item_extractor_fn = item_extractor_fn
        elif isinstance(item_extractor_fn, str):
            if item_extractor_fn == "zoomclf":
                self.item_extractor_fn = zoom_task_extractor
            elif item_extractor_fn == "zoomreg":
                self.item_extractor_fn = zoom_task_extractor
            elif item_extractor_fn == "rotation":
                self.item_extractor_fn = relative_rot_task_extractor
        else:
            self.item_extractor_fn = item_extractor_fn

        self.item_extractor_param_values = item_extractor_param_values

        self.X = []
        self.y = []

        with open(split_path) as file:
            for line in file:
                line_data = line.strip().split(" ")
                path = os.path.join(root, line_data[0])
                rgb_image_file_path = path.replace('***', 'rgb')
                depth_image_file_path = path.replace('***', 'depth')
                label = int(line_data[1])
                if os.path.exists(rgb_image_file_path) and os.path.exists(depth_image_file_path):
                    self.X.append((rgb_image_file_path, depth_image_file_path))
                    self.y.append(label)

    def shuffle(self):
        self.X, self.y = utils.shuffle(self.X, self.y, random_state=0)

    def __getitem__(self, index):
        images, object_label = self.X[index], self.y[index]
        rgb_image = pil_loader(images[0])
        depth_image = pil_loader(images[1])
        if self.pretext_task_type == "zoomclf":
            return self.item_extractor_fn(rgb_image, depth_image, object_label,
                                      self.item_extractor_param_values, task_type="clf")  # (rgb_image, depth_image, object_label), (pretext_task_rgb_image, pretext_task_depth_image, pretext_task_label)
        elif self.pretext_task_type == "zoomreg":
            return self.item_extractor_fn(rgb_image, depth_image, object_label,
                                      self.item_extractor_param_values, task_type="reg")
        else:
            return self.item_extractor_fn(rgb_image, depth_image, object_label,
                                                  self.item_extractor_param_values)

    def __len__(self):
        length = len(self.X)
        return length
