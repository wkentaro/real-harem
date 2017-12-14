#!/usr/bin/env python

import argparse
import io
import os.path as osp
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import requests
import skimage.io


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('img_file', nargs='?', default='../face2face/face1.jpg')
    args = parser.parse_args()

    if not osp.exists(args.img_file):
        print('File does not exist: %s' % args.img_file)
        return

    url = 'http://hoop.jsk.imi.i.u-tokyo.ac.jp'
    files = {'file': open(args.img_file, 'rb')}
    r = requests.post(url, files=files)
    img1 = skimage.io.imread(args.img_file)
    if r.status_code == 200:
        img2 = PIL.Image.open(io.BytesIO(r.content))
        img2 = np.asarray(img2)
        mask = (img2 != 0).all(axis=2)
        img1_dash = img1.copy()
        img1_dash[mask] = img2[mask]
        plt.subplot(121)
        plt.imshow(img1)
        plt.subplot(122)
        plt.imshow(img1_dash)
        plt.show()


if __name__ == '__main__':
    main()
