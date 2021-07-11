
import os
import logging
from typing import List, Tuple

import cv2
import pandas as pd
import numpy as np
from numpy.random import default_rng

from . import utils
from .. import const

if not os.path.exists('./logs'):
    os.makedirs('./logs')

logging.basicConfig(filename='./logs/dataset.log', level=logging.INFO)
logger = logging.getLogger(__name__)


class Dataset():
    def __init__(self,
                 size: tuple,
                 name: str = "",
                 d_type: str = const.DATASET,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        """Base class to interact with the original dataset to build custom datasets.

        Args:
            size (tuple): size of the output images in the dataset.
            name (str): Name of dataset used to build paths for
            storing generated files. Defaults to ''.
            d_type (str, optional): [description]. Defaults to 'dataset'.
            excludes (list, optional): [description]. Defaults to None.
            random_seed (int, optional): [description]. Defaults to None.
            rebuild (bool, optional): [description]. Defaults to False.
            save (bool, optional): Whether to save the generated files for
            faster builds in the future. Defaults to True.

        Raises:
            ValueError: Will fail to build on non-supported dataset type.
        """
        assert len(size) == 2
        self.size = size
        rows = self.size[0]
        cols = self.size[1]

        self.name = name
        self.d_type = d_type
        self.rng = default_rng(seed=random_seed)
        self.excludes = excludes

        # raw labels.xlsx
        self.ant_data = pd.read_excel(
            f'./dataset/labels.xlsx', header=0)

        self.img_meta_path = f'./dataset/{self.name}_{self.d_type}_{rows}_{cols}_img_meta.npy'
        self.images_path = f'./dataset/{self.name}_{self.d_type}_{rows}_{cols}_images.npy'
        self.labels_path = f'./dataset/{self.name}_{self.d_type}_{rows}_{cols}_labels.npy'
        self.ids_path = f'./dataset/{self.name}_{self.d_type}_{rows}_{cols}_ids.npy'

        # load img ids and labels
        self._build_labels(rebuild, save)
        logger.info('Loaded image metadata.')

        self._build_images(rebuild, save)
        logger.info(f'Loaded sub-image dataset of size {rows} by {cols}.')
        logger.info(
            f'Total of {len(self.labels)} sub-images built from {len(np.unique(self.ids))} images.')

        logger.info(f'Class data:')
        uniques = np.unique(self.labels, return_counts=True)
        for i in range(len(uniques[0])):
            logger.info(f'\t{uniques[0][i]}: {uniques[1][i]}')

    def _build_labels(self, rebuild: bool, save: bool):
        """Helper for Dataset.build_labels, it will use the rebuild variable to
        force a rebuild to new files. It will also rebuild on failure-to-load
        the expected files.

        Args:
            rebuild (bool): Force rebuilding the dataset.
            save (bool): Save the generated files.

        Updates:
            img_meta: img metadata used for getting sample info (label, etc.).
        """
        if rebuild:
            self.build_labels(save)

        try:
            self.img_meta = np.load(self.img_meta_path)
        except Exception as e:
            logger.debug(e)
            # build on failure-to-load
            self.img_meta = self.build_labels(save)

    def build_labels(self, save: bool) -> np.ndarray:
        """Builds the img_meta.npy file based on data from the labels.xlsx file.

        Args:
            save (bool): Save the generated files.
            dtype (str): If const.DATASET_RS, then convert the labels to 0 or 1.

        Returns:
            np.ndarray: Image metadata scraped from original dataset.
        """
        logger.info(f'Generating labels.')

        # convert human expert labels using majority voting
        label_cols = self.ant_data[['Jp', 'Becca', 'Katy']]
        if self.d_type == const.DATASET_RS:
            label = utils.convert_labels_rs(label_cols.mode(axis=1)[0])
        elif self.d_type == const.DATASET_ALL:
            label = utils.convert_labels(label_cols.mode(axis=1)[0])

        label['class'] = label['class'] + 1

        # clip between the number of classes
        m = int(label['class'].max())
        idx = label['class'].isin(range(1, m+1))
        label = label.loc[idx]

        id_col = self.ant_data['Photo_number']
        id_col = id_col.to_frame('id')
        id_col = id_col.loc[idx.values]

        # store as arrays
        img_labels = label['class'].values
        img_ids = id_col['id'].values

        assert len(img_labels) == len(img_ids)

        # save image ids and labels
        res = np.stack([img_ids.astype(int), img_labels.astype(int)])

        if save:
            np.save(self.img_meta_path, res)

        return res

    def _build_images(self, rebuild: bool, save: bool):
        """Helper for Dataset.build_images, it will use the rebuild variable to
        force a rebuild to new files. It will also rebuild on failure-to-load
        the expected files.

        Args:
            rebuild (bool): Force rebuilding the dataset.
            save (bool): Save the generated files.

        Updates:
            subimages: Array of generated images.
            labels: Ground-truth label of each image.
            ids: Original sample id of each image.
        """
        if rebuild:
            self.images, self.labels, self.ids = self.build_images(save)
            return

        try:
            self.images = np.load(self.images_path)
            self.labels = np.load(self.labels_path)
            self.ids = np.load(self.ids_path)
        except Exception as e:
            logger.debug(e)

            # build on failure-to-load
            self.images, self.labels, self.ids = self.build_images(save)

    def build_images(self, save: bool):
        'Must be implemented by the dataset type.'
        raise NotImplementedError

    def get_label(self, _id: int) -> int:
        """Given an image ID, return the label.

        Args:
            _id (int): ID of the original sample

        Raises:
            ValueError: NA label for iamge

        Returns:
            int: Original sample label
        """
        idx = np.where(self.img_meta[0] == _id)
        label = self.img_meta[1][idx]
        try:
            return int(label)
        except Exception as e:
            logger.debug(e)
            try:
                return int(label[0])
            except Exception as e:
                logger.debug(e)
                raise ValueError(
                    f'Class of ID[{_id}] is NA or not considered in this version.')

    def get_image(self, _id: int) -> np.ndarray:
        """Get image by ID.

        Args:
            _id (int): ID of the sample.

        Returns:
            img (np.ndarray): Image as cv2 image object (numpy array).
        """
        path = f'./dataset/data/{_id}.jpg'
        img = cv2.imread(path)

        if img is None:
            msg = f'Failed to open image {path}'
            logger.error(msg)
            raise ValueError(msg)

        return img

    def is_included(self, _id: int) -> bool:
        """Given an image id, return if the image is included from the dataset.

        Args:
            _id (int): ID of the sample.

        Returns:
            bool: True if included, else false
        """
        if _id in self.ids:
            return True
        return False

    def get_ant_info(self, _id: int) -> List[str]:
        """Get ant species info from original dataset.

        Args:
            _id (int): ID of the sample.

        Returns:
            List[str]: List of ant species info.
        """
        row = self.ant_data.loc[self.ant_data['Photo_number'] == _id]

        res = []
        if not row['Sub-species'].isnull().any():
            res.append(f'Sub-species: {row["Sub-species"].values[0]}')
        if not row['Species'].isnull().any():
            res.append(f'Species: {row["Species"].values[0]}')
        if not row['Subgenus'].isnull().any():
            res.append(f'Subgenus: {row["Subgenus"].values[0]}')
        if not row['Genus'].isnull().any():
            res.append(f'Genus: {row["Genus"].values[0]}')
        if not row['Sub-Family'].isnull().any():
            res.append(f'Sub-Family: {row["Sub-Family"].values[0]}')

        return res

    def stratified_split(self, n: int) -> Tuple[np.ndarray, np.ndarray]:
        """Stratified split with $n samples per class.

        Args:
            n (int): Number of samples per class

        Returns:
            train_x: Training images
            train_y: Training labels
            test_x: Test images
            test_y: Training labels
        """
        labels = self.labels-1
        uniques = np.unique(labels, return_counts=True)

        # stratified sample n for each class in self.labels
        train_idxs = np.zeros((len(uniques[0])*n), dtype=np.int)
        for _class in range(len(uniques[0])):
            idx = np.where(labels == uniques[0][_class])[0]
            samples = self.rng.choice(idx, size=n, replace=False)
            train_idxs[_class*n:(_class+1)*n] = samples

        test_idxs = np.array([idx for idx in range(
            len(labels)) if idx not in train_idxs])

        self.train_x = self.images[(train_idxs)]
        self.train_y = labels[(train_idxs)]
        self.test_x = self.images[(test_idxs)]
        self.test_y = labels[(test_idxs)]

        assert len(self.train_y) + len(self.test_y) == len(self.labels)

        return self.train_x, self.train_y

    def build_validation_set(self, split: float = 0.5) -> Tuple[np.ndarray, np.ndarray]:
        try:
            # see if test_y exists
            self.train_y
        except Exception as e:
            msg = "Test samples have yet to be made."
            logger.error(msg)
            raise e

        val_idxs = self.rng.choice(
            np.arange(len(self.test_y)),
            size=int(split*len(self.test_y)),
            replace=False
        )

        test_idxs = np.array([idx for idx in range(
            len(self.test_y)) if idx not in val_idxs])

        self.test_x = self.test_x[(test_idxs)]
        self.test_y = self.test_y[(test_idxs)]
        self.val_x = self.test_x[(val_idxs)]
        self.val_y = self.test_y[(val_idxs)]

        return self.val_x, self.val_y

    def train_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.train_x, self.train_y

    def test_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.test_x, self.test_y

    def val_set(self) -> Tuple[np.ndarray, np.ndarray]:
        return self.val_x, self.val_y
