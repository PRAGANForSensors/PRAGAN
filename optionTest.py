import torch

device = torch.device('cuda:0')

test_blur_path = r'DataTest/input'
test_sharp_path = r'DataTest/target'
test_deblur_path = r'DataTest/result'

test_model_path = r'Model/gen.pkl'
test_model_path = r'Model/gen_checkpoint.pkl'