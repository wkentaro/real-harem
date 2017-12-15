#!/usr/bin/env python

import io

from flask import Flask
from flask import request
from flask import send_file
import numpy as np
import PIL.Image

from usbcam_stargan_transgender import Node

from lib import ndarray_to_binary

app = Flask(__name__)
node = Node(video=None)


@app.route('/', methods=['POST'])
def transgender():
    img = PIL.Image.open(request.files['file'].stream)
    img = np.asarray(img)
    if img.ndim == 2:
        img = np.tile(img[:, :, None], 3, axis=2)
    elif img.shape[2] == 4:
        # RGBA -> RGB
        img = img[:, :, :3]
    img.setflags(write=1)
    img2 = node.process(img, return_facemask=True)
    img_binary = ndarray_to_binary(img2)
    return send_file(io.BytesIO(img_binary), attachment_filename='image.jpg')


app.run('0.0.0.0', port=80)
