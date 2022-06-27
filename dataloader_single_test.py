#!/usr/bin/env python
# -*- coding: UTF-8 -*-
import os
import torch
from torch.utils import data
import numpy as np
from torchvision import transforms
import torchvision.transforms.functional as TF
from pdb import set_trace as stx
import random
from torch.utils.data import Dataset
from PIL import Image

class DataLoaderTest(Dataset):
    def __init__(self, inp_dir, mean=[.5, .5, .5], std=[.5, .5, .5]):
        super(DataLoaderTest, self).__init__()
        self.input_path = inp_dir
        self.input_path_image = os.listdir(inp_dir)

        self.inp_size = len(self.input_path_image)

        self.transforms = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return self.inp_size

    def __getitem__(self, index):

        path_inp = os.path.join(self.input_path, self.input_path_image[index])
        filename = os.path.splitext(os.path.split(path_inp)[-1])[0]

        inp = Image.open(path_inp).convert('RGB')
        inp = TF.to_tensor(inp)

        input = self.transforms(inp)

        return (input, filename)
