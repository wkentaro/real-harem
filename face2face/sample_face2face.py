#!/usr/bin/env python

import cv2
import dlib
import imutils
from imutils import face_utils
import numpy as np


def detect_face(detector, predictor, image):
    scale = 500. / image.shape[1]
    image = cv2.resize(image, None, None, fx=scale, fy=scale)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # detect faces in the grayscale image
    rects = detector(gray, 1)

    rect = rects[0]  # XXX
    shape = predictor(gray, rect)
    shape = face_utils.shape_to_np(shape)

    rect = face_utils.rect_to_bb(rect)

    rect = np.asarray(rect, dtype=np.float32)
    shape = np.asarray(shape, dtype=np.float32)
    rect /= scale
    shape /= scale

    return rect, shape


def draw_face_detection(image, rect, shape):
    # convert dlib's rectangle to a OpenCV-style bounding box
    # [i.e., (x, y, w, h)], then draw the face bounding box
    x, y, w, h = map(int, rect)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # show the face number
    cv2.putText(image, "Face", (x - 10, y - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # loop over the (x, y)-coordinates for the facial landmarks
    # and draw them on the image
    for (x, y) in shape:
        cv2.circle(image, (x, y), 1, (0, 0, 255), -1)

    return image


def polygons_to_mask(img_shape, polygons):
    import PIL.Image
    import PIL.ImageDraw
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    xy = list(map(tuple, polygons))
    PIL.ImageDraw.Draw(mask).polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def paste_face_to_face(img1, img2, rect1, rect2, shape1, shape2):

    def xywh_to_yx(xywh):
        x, y, w, h = xywh
        x1 = x
        x2 = x + w
        y1 = y
        y2 = y + h
        return y1, x1, y2, x2

    y1, x1, y2, x2 = map(int, xywh_to_yx(rect1))
    img1_roi = img1[y1:y2, x1:x2]
    src = shape1.copy()
    src[:, 0] -= x1
    src[:, 1] -= y1
    src[:, 0] = np.clip(src[:, 0], 0, img1_roi.shape[1])
    src[:, 1] = np.clip(src[:, 1], 0, img1_roi.shape[0])

    mask1 = np.hstack((
        np.arange(0, 17),
        np.arange(26, 21, -1),
        [27],
        np.arange(21, 16, -1),
    ))
    mask1 = shape1[mask1]
    mask1 = polygons_to_mask(img1.shape, mask1)
    y1, x1, y2, x2 = map(int, xywh_to_yx(rect1))
    mask1_roi = mask1[y1:y2, x1:x2]

    # y1, x1, y2, x2 = map(int, xywh_to_yx(rect2))
    dst = shape2.copy()
    # img2_roi = img2[y1:y2, x1:x2]

    import skimage.transform as tf
    import skimage.util
    tform3 = tf.ProjectiveTransform()
    tform3.estimate(dst, src)
    img1_roi_to2 = tf.warp(img1_roi, tform3, output_shape=img2.shape)
    img1_roi_to2 = skimage.util.img_as_ubyte(img1_roi_to2)
    mask1_roi_to2 = tf.warp(mask1_roi, tform3, output_shape=img2.shape)
    mask1_roi_to2 = skimage.util.img_as_bool(mask1_roi_to2)

    y1, x1, y2, x2 = map(int, xywh_to_yx(rect2))
    mask_copy = mask1_roi_to2[y1:y2, x1:x2]
    img2[y1:y2, x1:x2][mask_copy] = img1_roi_to2[y1:y2, x1:x2][mask_copy]

    return img2


def main():
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor('./shape_predictor_68_face_landmarks.dat')

    img1 = cv2.imread('./face1.jpg')
    img1 = imutils.resize(img1, width=500)
    rect1, shape1 = detect_face(detector, predictor, img1)

    img2 = cv2.imread('./face2.jpg')
    img2 = imutils.resize(img2, width=500)
    rect2, shape2 = detect_face(detector, predictor, img2)

    viz1 = img1.copy()
    draw_face_detection(viz1, rect1, shape1)

    viz2 = img2.copy()
    draw_face_detection(viz2, rect2, shape2)

    viz = paste_face_to_face(img1, img2, rect1, rect2, shape1, shape2)

    # visualization
    import matplotlib.pyplot as plt
    plt.subplot(131)
    plt.imshow(viz1[:, :, ::-1])
    plt.subplot(132)
    plt.imshow(viz2[:, :, ::-1])
    plt.subplot(133)
    plt.imshow(viz[:, :, ::-1])
    plt.show()


if __name__ == '__main__':
    main()
