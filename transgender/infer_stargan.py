#!/usr/bin/env python

import chainercv
import chainer_cyclegan
import numpy as np
import skimage.io

from usbcam_stargan_transgender import Node


d = chainer_cyclegan.datasets.CelebAStyle2StyleDataset('test', 'Male')


node = Node('stargan')
node2 = Node('stargan', male=False)

cols = []
for i in range(10):
    img_a, img_b = d[i]
    img_a = img_a.transpose(2, 0, 1)
    img_b = img_b.transpose(2, 0, 1)
    img_a = chainercv.transforms.center_crop(img_a, (128, 128))
    img_b = chainercv.transforms.center_crop(img_b, (128, 128))
    img_a = img_a.transpose(1, 2, 0)
    img_b = img_b.transpose(1, 2, 0)

    out_a = node.process_face(img_a)
    out_b = node2.process_face(img_b)

    col = [img_a, out_a, img_b, out_b]
    cols.append(np.vstack(col))

tile = np.hstack(cols)

skimage.io.imsave('infer_stargan.png', tile)
