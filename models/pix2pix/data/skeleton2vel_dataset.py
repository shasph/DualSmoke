"""Dataset class template

This module provides a template for users to implement custom datasets.
You can specify '--dataset_mode template' to use this dataset.
The class name should be consistent with both the filename and its dataset_mode option.
The filename should be <dataset_mode>_dataset.py
The class name should be <Dataset_mode>Dataset.py
You need to implement the following functions:
    -- <modify_commandline_options>:　Add dataset-specific options and rewrite default values for existing options.
    -- <__init__>: Initialize this dataset class.
    -- <__getitem__>: Return a data point and its metadata information.
    -- <__len__>: Return the number of images.
"""
import torchvision
from data.base_dataset import BaseDataset, get_transform
# from data.image_folder import make_dataset
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as TF
import torch.nn.functional as F
import glob
import numpy as np
import torch
import os
import math


class Skeleton2velDataset(BaseDataset):
    """A template dataset class for you to implement custom datasets."""
    @staticmethod
    def modify_commandline_options(parser, is_train):
        """Add new dataset-specific options, and rewrite default values for existing options.

        Parameters:
            parser          -- original option parser
            is_train (bool) -- whether training phase or test phase. You can use this flag to add training-specific or test-specific options.

        Returns:
            the modified parser.
        """
        parser.set_defaults(input_nc=1, output_nc=2)  # specify dataset-specific default values
        # parser.add_argument('--new_dataset_option', type=float, default=1.0, help='new dataset option')
        # parser.set_defaults(max_dataset_size=10, new_dataset_option=2.0)  # specify dataset-specific default values
        return parser

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions

        A few things can be done here.
        - save the options (have been done in BaseDataset)
        - get image paths and meta information of the dataset.
        - define the image transformation.
        """
        # save the option and dataset root
        BaseDataset.__init__(self, opt)
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        # get the image paths of your dataset;
        self.vel_paths = []
        self.label_paths = []
        # define the default transform function. You can use <base_dataset.get_transform>; You can also define your custom transform function
        self.transform = get_transform(opt)
        self.vel_paths = self._get_vel_paths(opt)
        self.label_paths = self._get_label_paths(opt)
        # self._check_dataset()

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index -- a random integer for data indexing

        Returns:
            a dictionary of data with their names. It usually contains the data itself and its metadata information.

        Step 1: get a random image path: e.g., path = self.image_paths[index]
        Step 2: load your data from the disk: e.g., image = Image.open(path).convert('RGB').
        Step 3: convert your data to a PyTorch tensor. You can use helpder functions such as self.transform. e.g., data = self.transform(image)
        Step 4: return a data point as a dictionary.
        """
        vel_path = self.vel_paths[index]
        # velocity = np.zeros((3, 256, 256), dtype=np.float32)
        velocity = np.zeros((2, 256, 256), dtype=np.float32)
        max_v_norm2d = -1
        with open(vel_path) as f:
            lines = f.readlines()
        for line in lines:
            columns = line.replace('[', ' ')
            columns = columns.replace(']', ' ')
            columns = columns.replace(',', ' ')
            columns = columns.replace('=', ' ')
            columns = columns.split()
            x = int(columns[0])
            y = int(columns[1])
            velocity[0, y, x] = float(columns[3]) 
            velocity[1, y, x] = float(columns[4]) 
            # velocity[2, y, x] = 0 
            v_norm2d = math.sqrt(pow(velocity[0, y, x], 2) + pow(velocity[1, y, x], 2))
            if max_v_norm2d < v_norm2d:
                max_v_norm2d = v_norm2d
            
        for ix in range(0, velocity.shape[2]):
            for iy in range(0, velocity.shape[1]):
                velocity[0, iy, ix] /= max_v_norm2d
                velocity[1, iy, ix] /= max_v_norm2d

        label_path = self.label_paths[index]
        label = Image.open(label_path).convert('L')
        label = TF.to_tensor(label)

        data_A = label
        data_B = torch.from_numpy(velocity)

        A_paths = f'{label_path}'    # needs to be a string
        B_paths = f'{vel_path}'
        return {'A': data_A, 'B': data_B, 'A_paths': A_paths, 'B_paths': B_paths}

    def __len__(self):
        """Return the total number of images."""
        return len(self.vel_paths)

    def _get_vel_paths(self, opt):
        v_paths = []
        for f in glob.glob(self.dir_AB + '/*'):
            vel = f + '/velocity.txt'
            v_paths.append(vel)
        return v_paths

    def _get_label_paths(self, opt):
        label_paths = []
        for l in glob.glob(self.dir_AB + '/*'):
            label = l + '/skeleton.png'
            label_paths.append(label)
        return label_paths
    
    

