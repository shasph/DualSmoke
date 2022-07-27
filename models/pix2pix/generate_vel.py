from numpy.lib.arraysetops import isin
import torch
from torchvision import transforms
from options.generator_options import GeneratorOptions
from data import create_dataset
from models import create_model
from PIL import Image

import argparse
import torchvision.transforms.functional as F
import os
import statistics
import numpy as np

"""
Generate command

lcs => vel (pix2pix)
    python generate_vel.py --dataroot ./ --name lcs2vel_pix2pix --model pix2pix --input_nc 1 --output_nc 2

wlcs => vel (pix2pix)
    python generate_vel.py --dataroot ./ --name wlcs2vel_pix2pix --model pix2pix --input_nc 1 --output_nc 2

wlcs => vel (ours)
    python generate_vel.py --dataroot ./ --name wlcs2vel_ours --model lcs2vel

Using dataset
lcs => vel(pix2pix)
    python generate_vel.py --dataroot ../dataset --name lcs2vel_pix2pix --model pix2pix --dataset_mode lcs2vel --input_nc 1 --output_nc 2

wlcs => vel(pix2pix)
    python generate_vel.py --dataroot ../dataset --name wlcs2vel_pix2pix --model pix2pix --dataset_mode wlcs2vel --input_nc 1 --output_nc 2

wlcs => vel(ours)
    python generate_vel.py --dataroot ../dataset --name wlcs2vel_ours --model lcs2vel --dataset_mode wlcs2vel
"""

# TODO:Use GMM
def save_lcs2npz(tensor, save_path):
    if not isinstance(tensor, torch.Tensor):
        return
    np_array = tensor.data[0].cpu().float().numpy().astype(np.float32)
    np_array = np_array.squeeze(0)
    thre = statistics.mode(np_array.flatten())
    img = (np_array > thre + 0.1) * 1
    npz_img = img[np.newaxis, :, :, np.newaxis]
    npz_img = npz_img.astype(np.float32)
    np.savez(save_path, npz_img)


# tensor:shape(ch, x, y) => npz:shape(z, y, x, ch)
def save_velocity2npz(tensor, save_path):
    if not isinstance(tensor, torch.Tensor):
        return
    np_array = tensor.data[0].cpu().float().numpy().astype(np.float32) # shape(ch, x, y)
    np_zeros = np.zeros((1, 256, 256), dtype=np.float32)
    np_array = np.concatenate([np_array, np_zeros], 0)
    np_array = np_array[:, :, :, np.newaxis] #shape(ch, x, y, z)
    np_array = np_array.transpose((3, 1, 2, 0))
    np.savez(save_path, np_array)

def save_velocity2txt(tensor, save_path):
    if not isinstance(tensor, torch.Tensor):
        return
    np_array = tensor.data[0].cpu().float().numpy()
    with open(save_path, 'w') as fout:
        for ix in range(0, np_array.shape[2]):
            for iy in range(0, np_array.shape[1]):
                fout.write(f'{ix} {iy} {np_array[0, iy, ix]} {np_array[1, iy, ix]}\n')

if __name__ == '__main__':
    opt = GeneratorOptions().parse()  # get test options
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

    if opt.use_dataset:
        dataset = create_dataset(opt)
        dataset_iter = iter(dataset)
        datas = next(dataset_iter)
        datas = next(dataset_iter)
    else:
        data_A_path = opt.dataroot
        data_A = Image.open(data_A_path).convert('L')
        data_A = F.to_tensor(data_A)
        data_A = data_A.unsqueeze(0)
        datas = {'A': data_A, 'A_paths': data_A_path}

    model.set_input(datas)  # unpack data from data loader
    model.test()           # run inference

    images = {'real_A':model.real_A, 'fake_B':model.fake_B }

    for label, image in images.items():
        # shape = (batch, CH, W, H)
        # when image is skeleton
        if label == 'real_A':
            save_lcs2npz(image, f'{result_dir}/LCS.npz')
            converted = transforms.ToPILImage(mode='L')((image[0] + 1) / 2)

        # when image is velocity-2D
        if label == 'fake_B':
            save_velocity2txt(image, f'{result_dir}/{label}.txt')
            save_velocity2npz(image, f'{result_dir}/{label}.npz')
            ch_zeros = torch.zeros((1, 1, 256, 256))
            concated = torch.concat((image.cpu(), ch_zeros), 1)
            converted = transforms.ToPILImage()((concated[0] + 1) / 2)
        converted.save(f'{result_dir}/{label}.png')
