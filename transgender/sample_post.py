#!/usr/bin/env python

import argparse
import io
import os.path as osp
import subprocess

import matplotlib.pyplot as plt
import numpy as np
import PIL.Image
import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--img-file', default='../face2face/face1.jpg')
    args = parser.parse_args()

    if not osp.exists(args.img_file):
        print('File does not exist: %s' % args.img_file)
        return

    url = 'http://hoop.jsk.imi.i.u-tokyo.ac.jp:5000'
    files = {'file': open(args.img_file, 'rb')}
    r = requests.post(url, files=files)
    if r.status_code == 200:
        img2 = PIL.Image.open(io.BytesIO(r.content))
        img2 = np.asarray(img2)
        plt.imshow(img2)
        plt.show()


if __name__ == '__main__':
    main()
