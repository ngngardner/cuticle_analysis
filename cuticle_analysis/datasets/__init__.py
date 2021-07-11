
from typing import List

import numpy as np

from .. import const

# register datasets
from .dataset import Dataset
from .full import FullDataset
from .sub import SubDataset
from .gabor import GaborRoughSmoothFull
from .gabor import GaborRoughSmoothSub


class AllFull(FullDataset):
    'Full sized image dataset with all original labels.'

    def __init__(self,
                 size: tuple,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        name = f'all_augmented'
        d_type = const.DATASET_ALL  # used for converting labels
        super().__init__(size,
                         name=name,
                         d_type=d_type,
                         excludes=excludes,
                         random_seed=random_seed,
                         rebuild=rebuild,
                         save=save)


class AllFullAugmented(FullDataset):
    'Full sized image dataset with all original labels and augmented data.'

    def __init__(self,
                 size: tuple,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        name = f'all_augmented'
        d_type = const.DATASET_ALL  # used for converting labels
        super().__init__(size,
                         name=name,
                         d_type=d_type,
                         excludes=excludes,
                         random_seed=random_seed,
                         rebuild=rebuild,
                         save=save)
        self.augment()

    def augment(self):
        self.images, self.labels = self.images, self.labels


class RoughSmoothFull(FullDataset):
    'Full sized image dataset with rough and smooth labels only.'

    def __init__(self,
                 size: tuple,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        name = f'rs'
        d_type = const.DATASET_RS  # used for converting labels to rough or smooth
        super().__init__(size,
                         name=name,
                         d_type=d_type,
                         excludes=excludes,
                         random_seed=random_seed,
                         rebuild=rebuild,
                         save=save)


class RoughSmoothSub(SubDataset):
    'Subimage dataset with rough and smooth labels only.'

    def __init__(self,
                 size: tuple,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        name = f'rs'
        d_type = const.DATASET_RS  # used for converting labels to rough or smooth
        super().__init__(size,
                         name=name,
                         d_type=d_type,
                         excludes=excludes,
                         random_seed=random_seed,
                         rebuild=rebuild,
                         save=save)
