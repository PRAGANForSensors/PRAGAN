import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR, ExponentialLR, CosineAnnealingLR
import torch.optim as optim
import numpy as np
import os

from torch.utils.data import DataLoader, Dataset

from torchvision import utils
from tqdm import tqdm

from generator import *
from discriminator import *
from Net.ViTRealClass import *
from dataloader_train import *
from msssim import *
from Edge import *
from warmup_scheduler.scheduler import *
from SelectData import *
from Seed import *

from optionTrain import *

seed_torch(1234)

model = MTRUV(3).to(device)

optimizer = optim.Adam(model.parameters(), lr=lr_rate)

scheduler_cos = CosineAnnealingLR(optimizer, epoch_nums, 1e-6)

scheduler_warmup = GradualWarmupScheduler(optimizer, multiplier=1, total_epoch=10, after_scheduler=scheduler_cos)

bceloss = nn.BCELoss().to(device)
mseloss = nn.MSELoss().to(device)
ssimloss = SSIM().to(device)
edgeloss = EdgeLoss(device).to(device)

fake_label = torch.tensor(0, dtype=torch.long, requires_grad=False).to(device)
real_label = torch.tensor(1, dtype=torch.long, requires_grad=False).to(device)

def main():
    torch.set_num_threads(3)
    model.train()

    vit = torch.load(r'Model/PreTrainViT.pkl', map_location='cpu')
    vit.to(device)
    optimizer_dis = optim.AdamW(vit.parameters(), lr=lr_rate_dis)
    vit.train()

    iter = 0

    random_last = 0
    random_epoch_num = 99999
    random_iter = 0
    random_list = [25, 10, 5, 2]

    select_random_double_fixed(rate=4, input_path_input=blur_path, input_path_target=sharp_path,
                               target_path_input=select_blur_path, target_path_target=select_sharp_path)
    train_data = MyDataset(select_blur_path, select_sharp_path, image_size=image_size)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

    for epoch in tqdm(range(epoch_nums)):

        select_flag = False
        if epoch in epochs_list:
            select_random_double_fixed(rate=rate, input_path_input=blur_path, input_path_target=sharp_path,
                                           target_path_input=select_blur_path, target_path_target=select_sharp_path)
            random_last = epoch
            random_epoch_num = random_list[random_iter]
            random_iter += 1
            select_flag = True

        elif (epoch - random_last) == random_epoch_num:
            random_last = epoch
            select_random_double_fixed(rate=rate, input_path_input=blur_path, input_path_target=sharp_path,
                                           target_path_input=select_blur_path, target_path_target=select_sharp_path)
            select_flag = True

        if select_flag:
            train_data = MyDataset(select_blur_path, select_sharp_path, image_size=image_size)
            train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, drop_last=True)

        for idx, (blur, sharp) in enumerate(train_loader):

            blur = blur.to(device)
            sharp = sharp.to(device)

            with torch.no_grad():
                model.eval()
                res1, hidden, cell = model(blur)

                res2, hidden, cell = model(blur, hidden, cell)

                res3, hidden, cell = model(blur, hidden, cell)

            vit.zero_grad()
            real = vit(sharp).mean()
            fake_1 = vit(res1).mean()
            fake_2 = vit(res2).mean()
            fake_3 = vit(res3).mean()
            fake = (fake_1 + fake_2 + fake_3) / 3
            # ---------------------------------
            loss_dis = 0.5 * (bceloss(fake, fake_label.float()) + bceloss(real, real_label.float()))
            # ---------------------------------
            loss_dis.backward(retain_graph=True)
            optimizer_dis.step()

            model.train()
            model.zero_grad()

            res1, hidden, cell = model(blur)

            res2, hidden, cell = model(blur, hidden, cell)

            res3, hidden, cell = model(blur, hidden, cell)

            loss1 = mseloss(res1, sharp)
            loss2 = edgeloss(res2, sharp)
            loss3 = 1 - ssimloss(res3, sharp)

            loss = loss1 + loss2 + loss3 + bceloss(fake, real_label.float()).detach()
            loss.backward()
            optimizer.step()

            iter += 1

            if (idx + 1) % 5 == 0:
                print('\rEpoch [{}/{}], Step {}, MSE: {:.4f}, Edge: {:.4f}, SSIM: {:.4f}, DIS:{:.4f}'
                      .format(epoch + 1, epoch_nums, idx, loss1, loss2, 1 - loss3, loss_dis), end='')  # {}里面是后面需要传入的变量

        scheduler_warmup.step()

        if (epoch % 50) == 0:

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optim_state_dist': optimizer.state_dict()
            }
            checkpoint_dis = {
                'epoch': epoch,
                'model_state_dict': vit.state_dict(),
                'optim_state_dist': optimizer_dis.state_dict()
            }
            torch.save(checkpoint_dis, str(model_save_path + "/dis.pkl"))
            torch.save(checkpoint, str(model_save_path + "/gen.pkl"))


if __name__ == '__main__':
    main()