# flask etc.
from flask import Flask, request
from flask.helpers import make_response
from flask.wrappers import Response
from sklearn.mixture import GaussianMixture
from scipy import ndimage
import subprocess
import os
import shutil
import argparse
import sys
import time

# pytorch etc.
import torch
from torch.autograd import Variable
from torchvision import transforms
from options.generator_options import GeneratorOptions
from data import create_dataset
from models import create_model
from PIL import Image, ImageOps
import torchvision.transforms.functional as F
import numpy as np
import statistics
import collections


app = Flask(__name__)
model_sketch2lcs = None
model_lcs2vel = None
result_dir = 'http/tmp'

@app.route('/hello', methods=['GET'])
def hello():
    return Response(response='Successful Connection')

@app.route('/upload', methods=['POST'])
def upload():
    img = request.files['sketch.png']
    img.save('http/sketch.png')
    return Response(response='Upload done.', status=200)

@app.route('/generate', methods=['POST'])
def generate():
    os.makedirs(result_dir, exist_ok=True)
    generate_sketch2lcs()
    generate_lcs2vel(generate_img=True)
    shutil.make_archive('./result', 'zip', root_dir=result_dir)
    shutil.rmtree(result_dir)

    response = make_response()
    response.data = open('./result.zip', 'rb').read()
    response.headers['Content-Type'] = 'application/octet-stream'
    response.headers['Content-Disopsition'] = 'attachment; filename=result.zip'
    return response

def setup_sketch2lcs_model():
    global model_sketch2lcs
    argv_origin = sys.argv
    dataroot='http/sketch.png'
    cmd_argv = f'null --dataroot {dataroot} --name skeleton2lcs --model pix2pix --input_nc 1 --output_nc 1 --result_dir {result_dir}'
    cmd_argv_list = cmd_argv.split()
    sys.argv = cmd_argv_list
    opt = GeneratorOptions().parse()
    sys.argv = argv_origin
    model_sketch2lcs = create_model(opt)
    model_sketch2lcs.setup(opt)

def setup_lcs2vel_model():
    global model_lcs2vel
    argv_origin = sys.argv
    dataroot = f'{result_dir}/LCS.png'
    cmd_argv = f'null --dataroot {dataroot} --name lcs2vel_pix2pix --model pix2pix --input_nc 1 --output_nc 2 --result_dir {result_dir}'
    cmd_argv_list = cmd_argv.split()
    sys.argv = cmd_argv_list
    opt = GeneratorOptions().parse()
    sys.argv = argv_origin
    model_lcs2vel = create_model(opt)
    model_lcs2vel.setup(opt)

def generate_sketch2lcs():
    data_A_path = 'http/sketch.png'
    data_A = Image.open(data_A_path).convert('L')
    data_A = F.to_tensor(data_A)
    data_A = data_A.unsqueeze(0)
    datas = {'A': data_A, 'A_paths': data_A_path}

    sketch2lcs_time = time.time()
    fake_B = generate(model_sketch2lcs, datas)
    sketch2lcs_time = time.time() - sketch2lcs_time
    print(f'Sketch2Lcs : {sketch2lcs_time}')

    converted = transforms.ToPILImage(mode='L')((fake_B[0] + 1) / 2)
    converted.save(f'{result_dir}/LCS.png')
    np_array = fake_B.data[0].cpu().float().numpy().astype(np.float32)
    np_array = np_array.squeeze(0)
    # thre = collections.Counter(np_array.flatten()).most_common()[0][0]
    im = ndimage.gaussian_filter(np_array, sigma=2.5, mode='constant', cval=0)
    classif = GaussianMixture(n_components=2)
    classif.fit((im.reshape(im.size, 1)))
    thre = -10
    for mean in classif.means_:
        if(mean > thre):
            thre = mean
    img = (np_array > thre) * 1
    npz_img = img[np.newaxis, :, :, np.newaxis]
    npz_img = npz_img.astype(np.float32)
    np.savez(f'{result_dir}/LCS.npz', npz_img)

def generate_lcs2vel(generate_img=False):
    data_A_path = 'http/tmp/LCS.png'
    data_A = Image.open(data_A_path).convert('L')
    data_A = F.to_tensor(data_A)
    data_A = data_A.unsqueeze(0)
    datas = {'A': data_A, 'A_paths': data_A_path}

    lcs2vel_time = time.time()
    fake_B = generate(model_lcs2vel, datas)
    lcs2vel_time = time.time() - lcs2vel_time
    print(f'Lcs2Vel : {lcs2vel_time}')

    if generate_img:
        ch_zeros = torch.zeros((1, 1, 256, 256))
        concated = torch.concat((fake_B.cpu(), ch_zeros), 1)
        converted = transforms.ToPILImage()((concated[0] + 1) / 2)
        converted.save(f'{result_dir}/Vel.png')
    np_array = fake_B.data[0].cpu().float().numpy().astype(np.float32) # shape(ch, x, y)
    np_zeros = np.zeros((1, 256, 256), dtype=np.float32)
    np_array = np.concatenate([np_array, np_zeros], 0)
    np_array = np_array[:, :, :, np.newaxis] #shape(ch, x, y, z)
    np_array = np_array.transpose((3, 1, 2, 0))
    np.savez(f'{result_dir}/Vel.npz', np_array)

def generate(model, datas):
    model.set_input(datas)
    model.test()
    return model.fake_B

if __name__=='__main__':
    setup_sketch2lcs_model()
    setup_lcs2vel_model()
    app.run(debug=True, host='localhost', port=8080)
