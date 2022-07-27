import numpy as np
import os
import ntpath
import csv

def save_image(image, save_path, len_x, len_y):
    nd_img = image[0].cpu().float().numpy()
    with open(save_path, 'w') as fout:
        for ix in range(0, len_x):
            for iy in range(0, len_y):
                fout.write(f"{ix} {iy} {0} {nd_img[0, iy, ix]} {nd_img[1, iy, ix]} 0\n")

def save_velocity_field(webpage, visuals, image_path):
    image_dir = webpage.get_image_dir()
    short_path = ntpath.basename(image_path[0])
    name = os.path.splitext(short_path)[0]

    for label, im_data in visuals.items():
        image_name = '%s_%s.txt' % (name, label) 
        save_path = os.path.join(image_dir, image_name)
        save_image(im_data, save_path, 256, 256)
    
def write_loss(file, losses):
    if os.path.isfile(file):
        isInit = True
    else:
        isInit = False

    with open(file, 'a') as f:
        writer = csv.DictWriter(f, losses.keys())
        if not isInit:
            writer.writeheader()
        writer.writerow(losses)
