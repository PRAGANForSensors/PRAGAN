
import math
import os
import cv2
import numpy as np
from numpy import *
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity
import csv
import codecs

def data_write_csv(file_name, datas):
    file_csv = codecs.open(file_name, 'w+', 'utf-8')
    writer = csv.writer(file_csv, delimiter=' ', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
    for data in datas:
        writer.writerow(data)

def storFile(fileName, data):
    with open(fileName,'w',newline ='') as f:
        mywrite = csv.writer(f)
        mywrite.writerow(data)

def metric(input_path, target_path, score_path):
    images = os.listdir(target_path)
    psnr = []
    ssim = []
    for img in images:

        input = cv2.imread(os.path.join(input_path, img), cv2.COLOR_BGR2YCR_CB)
        target = cv2.imread(os.path.join(target_path, img), cv2.COLOR_BGR2YCR_CB)

        if input is None:
            img = img.split('.')[0]+'.png'
            input = cv2.imread(os.path.join(input_path, img), cv2.COLOR_BGR2YCR_CB)

        if target is None:
            img = img.split('.')[0]+'.jpg'
            target = cv2.imread(os.path.join(target_path, img), cv2.COLOR_BGR2YCR_CB)

        psnr.append(peak_signal_noise_ratio(input, target))
        ssim.append(structural_similarity(input, target, multichannel=True))
        # print('PSNR:{}'.format(psnr)+' ,SSIM:{}'.format(ssim))

    psnr_avg = mean(psnr)
    ssim_avg = mean(ssim)

    print('PSNR:{:.2f}/SSIM:{:.3f}'.format(psnr_avg, ssim_avg))

    storFile(score_path+ "/PSNR_{}.csv".format(psnr_avg), psnr)
    storFile(score_path + "/SSIM_{}.csv".format(ssim_avg), ssim)


if __name__ == '__main__':
    res_path = r'DataTest/result'
    tar_path = r'DataTest/target'
    sco_path = r'DataTest/score'

    metric(res_path, tar_path, sco_path)