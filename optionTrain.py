import torch

device = torch.device('cuda:0')
lr_rate = 2e-4
lr_rate_dis = 1e-5
batch_size = 2
epoch_nums = 251
image_size = 256

blur_path = r'DataTrain/input'
sharp_path = r'DataTrain/target'
select_blur_path = r'DataTrain/Select/input'
select_sharp_path = r'DataTrain/Select/target'
model_save_path = r'Model'

epochs_list = [epoch_nums//5, epoch_nums//5*2, epoch_nums//5*3, epoch_nums//5*4]
random_list = [25, 10, 5, 2]
rate = 4