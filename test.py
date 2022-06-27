
import torch
import os
from torch.utils.data import DataLoader, Dataset
# from torchvision.utils import save_image
from tqdm import tqdm

from generator import *
from dataloader_single_test import *
from imageUtil import *

from optionTest import *

test_data = DataLoaderTest(test_blur_path)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False)
model = MTRUV(3).to(device)
checkpoint = torch.load(test_model_path, map_location='cpu')  # .to(device)
model.load_state_dict(checkpoint['model_state_dict'])

def test():

    torch.set_num_threads(3)

    os.makedirs(test_deblur_path, exist_ok=True)

    for ii, (blur, name) in tqdm(enumerate(test_loader)):
        blur = blur.to(device)

        with torch.no_grad():

            res1, hidden, cell = model(blur)
            #
            res2, hidden, cell = model(blur, hidden, cell)
            # # #
            res3, hidden, cell = model(blur, hidden, cell)

            name = str(name).split('(\'')
            name = name[1].split('\',)')

            imageSavePIL(res3, name[0]+'.png', save_path=test_deblur_path)


if __name__ == '__main__':
    test()