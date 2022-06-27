#!/usr/bin/env python
# -*- coding: UTF-8 -*-

from PIL import Image
import os
from tqdm import tqdm
import random
from random import random, uniform, randint, randrange, choice, sample, shuffle

def select_random_double(select_num, in_path_input, in_path_target, save_path_input, save_path_target):
    i = 0
    total_num = 0
    total = []
    all_images = os.listdir(in_path_input)
    last_select = os.listdir(save_path_input)

    for image in all_images:
        total_num = total_num+1

    for j in range(total_num):
        total.append(i)
        i = i+1

    select = (sample(total, select_num))
    i = 0

    # for image in tqdm(all_images):
    for select_last in last_select:
        os.remove(os.path.join(save_path_input, select_last))
        os.remove(os.path.join(save_path_target, select_last))

    for image in all_images:
        if i in select:
            image_path = os.path.join(in_path_input, image)
            img = Image.open(image_path)
            img.save(save_path_input + '/' + image)

            image_path = os.path.join(in_path_target, image)
            img = Image.open(image_path)
            img.save(save_path_target + '/' + image)

        i = i + 1

def countNum(path):
    all_the_images = os.listdir(path)
    i = 0
    for image in all_the_images:
        i = i+1
    return i

def select_random_double_fixed(rate, input_path_input, input_path_target, target_path_input, target_path_target):

    i = 0
    total_num = 0
    total = []
    all_images = os.listdir(input_path_input)
    last_select = os.listdir(target_path_input)

    for image in all_images:
        total_num = total_num+1

    for j in range(total_num):
        total.append(i)
        i = i+1

    select = (sample(total, countNum(input_path_target)//rate))
    i = 0

    # for image in tqdm(all_images):
    for select_last in last_select:
        os.remove(os.path.join(target_path_input, select_last))
        os.remove(os.path.join(target_path_target, select_last))

    for image in all_images:
        if i in select:
            image_path = os.path.join(input_path_input, image)
            img = Image.open(image_path)
            img.save(target_path_input + '/' + image)

            image_path = os.path.join(input_path_target, image)
            img = Image.open(image_path)
            img.save(target_path_target + '/' + image)

        i = i + 1


