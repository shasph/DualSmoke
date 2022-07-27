from PIL import Image, ImageOps
import numpy as np
import settings
import math

img = np.zeros((256, 256))

with open('./tmp/obs.txt', 'r') as fr:
    lines = fr.readlines()

# 一行目に解像度(X,Y)
srcResolution = int(lines[0].split()[0])
dstResolution = settings.RESOLUTION

iter_lines = iter(lines)
next(iter_lines)

for line in iter_lines:
    data = line.split()
    x = float(data[0])
    y = float(data[1])
    x = math.floor(x / srcResolution * dstResolution)
    y = math.floor(y / srcResolution * dstResolution)
    img[int(y), int(x)] = 255

pil_image = Image.fromarray(img).convert('L')
pil_image = ImageOps.invert(pil_image)
pil_image.save('sketch.png')