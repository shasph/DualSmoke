import torch
from torchvision import transforms
from options.generator_options import GeneratorOptions
from data import create_dataset
from models import create_model
from PIL import Image, ImageOps

import torchvision.transforms.functional as F
import os

"""
Generate command
skeleton => lcs
    python generate_lcs.py --dataroot ./ --name skeleton2lcs --model pix2pix --input_nc 1 --output_nc 1

skeleton => wlcs
    python generate_lcs.py --dataroot ./ --name skeleton2wlcs --model pix2pix --input_nc 1 --output_nc 1

Using dataset
    python generate_lcs.py --dataroot ../dataset --name skeleton2lcs --model pix2pix --dataset_mode skeleton2lcs --input_nc 1 --output_nc 1

    python generate_lcs.py --dataroot ../dataset --name skeleton2wlcs --model pix2pix --dataset_mode skeleton2wlcs --input_nc 1 --output_nc 1
"""

if __name__ == '__main__':
    opt = GeneratorOptions().parse()  # get test options
    print(opt)
    # hard-code some parameters for test
    opt.num_threads = 0   # test code only supports num_threads = 0
    opt.batch_size = 1    # test code only supports batch_size = 1
    opt.serial_batches = False  # disable data shuffling; comment this line if results on randomly chosen images are needed.
    opt.no_flip = False    # no flip; comment this line if results on flipped images are needed.
    opt.display_id = -1   # no visdom display; the test code saves the results to a HTML file.
    model = create_model(opt)      # create a model given opt.model and other options
    model.setup(opt)               # regular setup: load and print networks; create schedulers

    result_dir = opt.result_dir
    os.makedirs(result_dir, exist_ok=True)
    
    if opt.eval:
        model.eval()
    data_A_path = './generator/skeleton/sketch.png'
    data_A = Image.open(data_A_path).convert('L')
    data_A = F.to_tensor(data_A)
    data_A = data_A.unsqueeze(0)
    datas = {'A': data_A, 'A_paths': data_A_path}

    # dataset = create_dataset(opt)
    # dataset_iter = iter(dataset)
    # datas = next(dataset_iter)
    # datas = next(dataset_iter)
    # print(datas['A_paths'])

    model.set_input(datas)  # unpack data from data loader
    model.test()           # run inference

    images = {'real_A':model.real_A, 'fake_B':model.fake_B }

    for label, image in images.items():
        converted = transforms.ToPILImage(mode='L')((image[0] + 1) / 2)
        converted.save(f'{result_dir}/{label}.png')