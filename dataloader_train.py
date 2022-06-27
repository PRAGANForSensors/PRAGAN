
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


class MyDataset(data.Dataset):
    def __init__(self, input_path, label_path, image_size,  mean=[.5, .5, .5], std=[.5, .5, .5]):
        self.image_size = image_size
        self.input_path = input_path
        self.input_path_image = os.listdir(input_path)

        self.label_path = label_path
        self.label_path_image = os.listdir(label_path)

        self.transforms = transforms.Compose([
            transforms.Normalize(mean=mean, std=std)
        ])

    def __len__(self):
        return len(self.input_path_image)

    def __getitem__(self, index):
        ps = self.image_size

        input_image_path = os.path.join(self.input_path, self.input_path_image[index])

        inp_img = Image.open(input_image_path).convert('RGB')

        label_image_path = os.path.join(self.label_path, self.label_path_image[index])
        tar_img = Image.open(label_image_path).convert('RGB')

        w, h = tar_img.size

        aug = random.randint(0, 2)
        if aug == 1:
            inp_img = TF.adjust_gamma(inp_img, 1)
            tar_img = TF.adjust_gamma(tar_img, 1)

        aug = random.randint(0, 2)
        if aug == 1:
            sat_factor = 1 + (0.2 - 0.4 * np.random.rand())
            inp_img = TF.adjust_saturation(inp_img, sat_factor)
            tar_img = TF.adjust_saturation(tar_img, sat_factor)

        inp_img = TF.to_tensor(inp_img)
        tar_img = TF.to_tensor(tar_img)

        hh, ww = tar_img.shape[1], tar_img.shape[2]

        rr = random.randint(0, hh - ps)
        cc = random.randint(0, ww - ps)
        aug = random.randint(0, 8)

        # Crop patch
        inp_img = inp_img[:, rr:rr + ps, cc:cc + ps]
        tar_img = tar_img[:, rr:rr + ps, cc:cc + ps]

        # Augmentations
        if aug == 1:
            inp_img = inp_img.flip(1)
            tar_img = tar_img.flip(1)
        elif aug == 2:
            inp_img = inp_img.flip(2)
            tar_img = tar_img.flip(2)
        elif aug == 3:
            inp_img = torch.rot90(inp_img, dims=(1, 2))
            tar_img = torch.rot90(tar_img, dims=(1, 2))
        elif aug == 4:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=2)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=2)
        elif aug == 5:
            inp_img = torch.rot90(inp_img, dims=(1, 2), k=3)
            tar_img = torch.rot90(tar_img, dims=(1, 2), k=3)
        elif aug == 6:
            inp_img = torch.rot90(inp_img.flip(1), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(1), dims=(1, 2))
        elif aug == 7:
            inp_img = torch.rot90(inp_img.flip(2), dims=(1, 2))
            tar_img = torch.rot90(tar_img.flip(2), dims=(1, 2))

        input = self.transforms(inp_img)
        label = self.transforms(tar_img)

        return (input, label)