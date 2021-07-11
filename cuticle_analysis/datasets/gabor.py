

from typing import List

import cv2
import numpy as np

from .. import const

from .full import FullDataset
from .sub import SubDataset


def build_filters() -> list:
    filters = []
    ksize = 31
    for theta in np.arange(0, np.pi, np.pi / 8):
        kern = cv2.getGaborKernel(
            (ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters


def process(img, kern):
    accum = np.zeros_like(img)
    fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
    np.maximum(accum, fimg, accum)
    return accum


class GaborRoughSmoothFull(FullDataset):
    """Full sized image dataset with rough and smooth labels only with added 
    gabor filter on preprocessing step.
    """

    def __init__(self,
                 size: tuple,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        name = f'gabor_rs'
        d_type = const.DATASET_RS  # used for converting labels to rough or smooth
        super().__init__(size,
                         name=name,
                         d_type=d_type,
                         excludes=excludes,
                         random_seed=random_seed,
                         rebuild=rebuild,
                         save=save)

    def preprocess(self, img: np.ndarray) -> List[np.ndarray]:
        'Override Dataset.preprocess to add gabor filter to each image'
        imgs = []

        filters = build_filters()
        for kern in filters:
            imgs.append(process(img, kern))

        return imgs


class GaborRoughSmoothSub(SubDataset):
    """Subimage dataset with rough and smooth labels only with added gabor 
    filter on preprocessing step.
    """

    def __init__(self,
                 size: tuple,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        name = f'gabor_rs'
        d_type = const.DATASET_RS  # used for converting labels to rough or smooth
        super().__init__(size,
                         name=name,
                         d_type=d_type,
                         excludes=excludes,
                         random_seed=random_seed,
                         rebuild=rebuild,
                         save=save)

    def preprocess(self, img: np.ndarray) -> List[np.ndarray]:
        'Override Dataset.preprocess to add gabor filter to each image'
        imgs = []

        filters = build_filters()
        for kern in filters:
            imgs.append(process(img, kern))

        return imgs
