#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import torch
from torchvision import utils
from torchvision import transforms
import os
from PIL import Image
import random
from random import random, uniform, randint, randrange, choice, sample, shuffle
# from optionsSin import *

def imageSavePIL(images, fileName, save_path, normalization=True, mean=[.5, .5, .5], std=[.5, .5, .5]):
    image = utils.make_grid(images.cpu().detach())

    if normalization:

        image = image.permute(1, 2, 0)
        image = (image * torch.tensor(std) + torch.tensor(mean))

        image = image.permute(2, 0, 1)

    image = transforms.ToPILImage()(image)

    image.save(save_path + "/" + fileName)

def countNum(path):
    all_the_images = os.listdir(path)
    i = 0
    for image in all_the_images:
        i = i+1
    return i

def select_random(select_num, in_path, save_path):
    i = 0
    total_num = 0
    total = []
    all_images = os.listdir(in_path)
    last_select = os.listdir(save_path)

    for image in all_images:
        total_num = total_num+1

    for j in range(total_num):
        total.append(i)
        i = i+1

    select = (sample(total, select_num))
    i = 0
    all_images = os.listdir(in_path)
    # for image in tqdm(all_images):
    for select_last in last_select:
        os.remove(os.path.join(save_path, select_last))

    for image in all_images:
        if i in select:
            image_path = os.path.join(in_path, image)
            img = Image.open(image_path)
            img.save(save_path + '/' + image)

        i = i + 1


