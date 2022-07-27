import numpy as np
import settings
from PIL import Image, ImageOps

def rasterize():
    dstResolution = settings.RESOLUTION
    with open('tmp/obs.txt', 'r') as fin:
        datalist = fin.readlines()

    srcResolution = int(datalist[0].split(' ')[0])
    src = []

    for line in datalist:
        string = line.split(' ')
        src.append([float(string[0]), float(string[1])])

    dst = np.zeros((1, dstResolution, dstResolution, 1), dtype='int32')
    q = int(srcResolution / dstResolution)

    for n in range(1, len(src)):
        x = int(src[n][0] / q)
        y = int(src[n][1] / q)
        # 通常
        # dst[0, y, x, 0] = 2
        # 上下反転
        dst[0, dstResolution - y, x, 0] = 2

    np.savez('tmp/obs', dst)
    dst = dst.astype(np.uint8) / 2 * 255
    dst = np.squeeze(dst)
    dst_img = Image.fromarray(dst).convert('L')
    dst_img = ImageOps.invert(dst_img)
    dst_img.save('tmp/sketch.png')