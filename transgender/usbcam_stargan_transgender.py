#!/usr/bin/env python

import os.path as osp

import numpy as np
import torch
from torch.backends import cudnn
from torch.autograd import Variable

cudnn.benchmark = True

import cv2
import dlib

here = osp.dirname(osp.realpath(__file__))
stargan_dir = osp.realpath(osp.join(here, 'StarGAN'))

import sys
sys.path.insert(0, stargan_dir)
from solver import Generator

sys.path.insert(0, osp.join(here, '../face2face'))
from sample_face2face import paste_face_to_face
from sample_face2face import face_utils


class Node(object):

    def __init__(self, video=0):
        G_path = osp.join(stargan_dir, 'stargan_celebA/models/20_4000_G.pth')
        G = Generator(64, c_dim=5, repeat_num=6)
        G.load_state_dict(torch.load(G_path))
        G.eval()
        G.cuda()
        self.G = G

        if video is not None:
            self.video = cv2.VideoCapture(video)

        self.detector = dlib.get_frontal_face_detector()
        self.predictor = dlib.shape_predictor(osp.join(here, '../face2face/shape_predictor_68_face_landmarks.dat'))

    def mainloop(self):
        while True:
            ret, img = self.video.read()
            img = img[:, :, ::-1]  # BGR -> RGB

            try:
                img2 = self.process(img)
            except Exception as e:
                print(e)
                continue
            viz = np.hstack([img, img2])
            cv2.imshow('img2', viz[:, :, ::-1])
            cv2.waitKey(1)

    def process(self, img, return_facemask=False):
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        dets = self.detector(img, 1)

        print('%d Faces are detected.' % len(dets))

        if return_facemask:
            img2_final = np.zeros_like(img)
        else:
            img2_final = img.copy()

        img2 = img.copy()
        img_H, img_W = img.shape[:2]
        for d in dets:
            rect = face_utils.rect_to_bb(d)
            shape = self.predictor(img, d)
            shape = face_utils.shape_to_np(shape)

            x1 = min(max(d.left(), 0), img_W)
            x2 = min(max(d.right(), 0), img_W)
            y1 = min(max(d.top(), 0), img_H)
            y2 = min(max(d.bottom(), 0), img_H)

            # enlarge bbox
            cx = (x1 + x2) / 2.
            cy = (y1 + y2) / 2.
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            bbox_w *= 1.8
            bbox_h *= 1.8
            cy -= (bbox_h * 0.15)
            x1 = cx - (bbox_w / 2.)
            x2 = x1 + bbox_w
            y1 = cy - (bbox_h / 2.)
            y2 = y1 + bbox_h
            x1 = min(max(x1, 0), img_W)
            x2 = min(max(x2, 0), img_W)
            y1 = min(max(y1, 0), img_H)
            y2 = min(max(y2, 0), img_H)
            y1, x1, y2, x2 = map(int, [y1, x1, y2, x2])

            cv2.rectangle(img2_final, (x1, y1), (x2, y2), (0, 0, 255), 2)

            # normalize
            xi = img[y1:y2, x1:x2].copy()
            xi = cv2.resize(xi, (128, 128))
            xi = xi.astype(np.float32) / 255.
            xi = xi * 2 - 1
            # img -> net input
            xi = xi.transpose(2, 0, 1)
            x = xi[None, :, :, :]
            x = torch.from_numpy(x)
            if torch.cuda.is_available():
                x = x.cuda()
            x = Variable(x, volatile=True)

            c = np.array([[1, 0, 0, 0, 1]], dtype=np.float32)
            c = torch.from_numpy(c)
            if torch.cuda.is_available():
                c = c.cuda()

            y = self.G(x, c)

            yi = y[0]
            yi = yi.data.cpu()
            yi = (yi + 1) / 2
            yi = yi.clamp_(0, 1)
            yi = yi.numpy()
            yi = (yi * 255).astype(np.uint8)
            yi = yi.transpose(1, 2, 0)

            roi_H, roi_W = y2 - y1, x2 - x1
            yi = cv2.resize(yi, (roi_W, roi_H))

            img2[y1:y2, x1:x2] = yi

            paste_face_to_face(img2, img2_final, rect, rect, shape, shape)

        return img2_final


if __name__ == '__main__':
    app = Node()
    # img = cv2.imread('StarGAN/face2016/k-okada_001.jpg')[:, :, ::-1]
    # img2 = app.process(img)
    # cv2 .imshow('a', img[:, :, ::-1])
    # cv2 .imshow('b', img2[:, :, ::-1])
    # cv2 .waitKey(0)
    app.mainloop()
