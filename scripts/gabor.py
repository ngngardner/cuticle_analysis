
import logging

import cv2
import numpy as np

from cuticle_analysis.datasets import RoughSmoothFull
from cuticle_analysis.models import CNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def build_filters():
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel(
            (ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    accum = np.zeros_like(img)
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


if __name__ == '__main__':
    import sys

    path = './dataset/data/1.jpg'
    img = cv2.imread(path)
    if img is None:
        print('Failed to load image file:', path)
        sys.exit(1)

    filters = build_filters()

    res = process(img, filters)

    cv2.imwrite('output/pre_gabor.jpg', img)
    cv2.imwrite('output/post_gabor.jpg', res)
