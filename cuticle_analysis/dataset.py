
import glob
import json
import logging
import os
import re
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import cv2
from labelme import utils
from numpy.random import default_rng

from . import const

if not os.path.exists('./logs'):
    os.makedirs('./logs')

logging.basicConfig(filename='./logs/dataset.log', level=logging.INFO)
logger = logging.getLogger(__name__)


def convert_labels(label: pd.Series):
    # replace label with 0 (rough) or 1 (smooth) based on first letter [r, s]
    label = label.replace(
        to_replace=r'^[r].*', value=0, regex=True)  # rough
    label = label.replace(
        to_replace=r'^[s].*', value=1, regex=True)  # smooth

    # remove the rest
    label = label.replace(
        to_replace=r'^[^rs].*', value=np.nan, regex=True)

    # convert to dataframe and filter by existing label
    label = label.to_frame('class')
    # label = label.loc[label['class'].isin([0, 1])]

    # increment to start index from 1 (images start from 1.jpg)
    label.index += 1

    return label


def get_label_names(img: np.ndarray, data: Dict) -> Tuple[np.ndarray, Dict]:
    # load segmented image data from json data
    label_name_to_value = {"_background_": 0}
    for shape in sorted(data["shapes"], key=lambda x: x["label"]):
        label_name = shape["label"]
        if label_name in label_name_to_value:
            label_value = label_name_to_value[label_name]
        else:
            label_value = len(label_name_to_value)
            label_name_to_value[label_name] = label_value
    lbl, _ = utils.shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    return lbl, label_names


class Dataset():
    def __init__(self,
                 size: tuple,
                 dataset_type: str = 'dataset',
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False):
        self.rng = default_rng(seed=random_seed)

        assert dataset_type in const.DATASETS
        self.dataset_type = dataset_type

        # raw labels.xlsx
        self.ant_data = pd.read_excel(
            f'./dataset/labels.xlsx', header=0)

        if not excludes:
            self.excludes = []
        else:
            self.excludes = excludes

        assert len(size) == 2
        self.size = size
        rows = self.size[0]
        cols = self.size[1]

        # for subimages dataset
        self.subimages_path = f'./dataset/{const.DS_MAP[dataset_type]}/{rows}_{cols}_img.npy'

        # for images dataset
        self.subimages_path = f'./dataset/{rows}_{cols}_img.npy'

        # for both
        self.labels_path = f'./dataset/{const.DS_MAP[dataset_type]}/{rows}_{cols}_labels.npy'
        self.ids_path = f'./dataset/{const.DS_MAP[dataset_type]}/{rows}_{cols}_ids.npy'
        self.img_meta_path = f'./dataset/{const.DS_MAP[dataset_type]}/img_meta.npy'

        # load img ids and labels
        self._build_labels(rebuild)
        logger.info('Loaded image metadata.')

        if dataset_type in const.SUBIMAGE_DATASETS:
            # load subimage dataset
            self._build_subimages(rebuild)
            logger.info(f'Loaded sub-image dataset of size {rows} by {cols}.')
            logger.info(
                f'Total of {len(self.labels)} sub-images built from {len(np.unique(self.ids))} images.')
        elif dataset_type == const.DATASET:
            self._build_images(rebuild)
            logger.info(f'Loaded image dataset of size {rows} by {cols}.')
            logger.info(f'Total of {len(np.unique(self.ids))} images.')
        else:
            raise ValueError(f'Dataset type {dataset_type} not handled')

        logger.info(f'Class data:')
        uniques = np.unique(self.labels, return_counts=True)
        for i in range(len(uniques[0])):
            logger.info(f'\t{uniques[0][i]}: {uniques[1][i]}')

    def _build_labels(self, rebuild: bool):
        if rebuild:
            self.build_labels()

        try:
            self.img_meta = np.load(self.img_meta_path)
        except Exception as e:
            logger.debug(e)
            # build on failure-to-load
            self.img_meta = self.build_labels()

    def build_labels(self) -> np.ndarray:
        """
            Builds the img_meta.npy file based on data from the labels.xlsx file.
        """
        logger.info(f'Generating labels.')

        # convert human expert labels using majority voting
        label_cols = self.ant_data[['Jp', 'Becca', 'Katy']]
        label = convert_labels(label_cols.mode(axis=1)[0])

        label['class'] = label['class'] + 1

        # clip between the number of classes
        idx = label['class'].isin(range(1, const.NUM_CLASSES+1))
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
        np.save(self.img_meta_path, res)

        return res

    def _build_subimages(self, rebuild: bool):
        if rebuild:
            self.subimages, self.labels, self.ids = self.build_subimages()
            return

        try:
            self.subimages = np.load(self.subimages_path)
            self.labels = np.load(self.labels_path)
            self.ids = np.load(self.ids_path)
        except Exception as e:
            logger.debug(e)

            # build on failure-to-load
            self.subimages, self.labels, self.ids = self.build_subimages()

    def build_subimages(self):
        """
            Builds the subimage dataset to store in .npy files.
        """
        logger.info(
            f'Generating dataset of size {self.size[0]} by {self.size[1]}.')

        # sub image data and original ids
        sub_images = {}
        sub_ids = {}

        if self.dataset_type in const.DS_MAP.keys():
            # if using rs or bg dataset, only use images with json
            files = glob.glob(
                f'./dataset/{const.DS_MAP[self.dataset_type]}/*.json')
        else:
            # else load each .jpg file
            files = glob.glob(f'./dataset/data/*.jpg')

        logger.info(f'Using {len(files)} images.')

        rows = self.size[0]
        cols = self.size[1]

        for file in files:
            # load image data based on avaiable jsons
            _id = int(re.findall('[0-9]+', file)[0])

            try:
                with open(file) as f:
                    data = json.load(f)

                img = cv2.imread(
                    f'./dataset/data/{_id}.jpg')
                img_label_name = self.get_label_name(_id)
                if img is None:
                    raise ValueError('Could not find image.')

                lbl, label_names = get_label_names(img, data)

                # add labels from label_names to the running sub_images data structures
                if label_names != list(sub_images.keys()):
                    for k in label_names:
                        if k not in sub_images.keys():
                            sub_images[k] = []
                            sub_ids[k] = []
                    if img_label_name not in sub_images.keys():
                        sub_images[img_label_name] = []
                        sub_ids[img_label_name] = []

                # pull 'cuticle data' from img as subimages
                for x in range(0, lbl.shape[0]-rows, rows):
                    for y in range(0, lbl.shape[1]-cols, cols):
                        idx = lbl[x:x+rows, y:y+cols]

                        # pull subimages that are composed entirely of one class
                        uniques = np.unique(idx)
                        if len(uniques) == 1:
                            # label of the subimage (background, etc...)
                            sub_label = uniques[0]

                            if self.dataset_type == const.DATASET_BG:
                                # only add background if there are less bg samples
                                # than cuticle
                                if label_names[sub_label] == '_background_':
                                    if len(sub_images['_background_']) \
                                            < len(sub_images['cuticle']):
                                        sub_images[label_names[sub_label]].append(
                                            img[x:x+rows, y:y+cols])
                                        sub_ids[label_names[sub_label]].append(
                                            _id)
                                else:
                                    sub_images[label_names[sub_label]].append(
                                        img[x:x+rows, y:y+cols])
                                    sub_ids[label_names[sub_label]].append(_id)

                            elif self.dataset_type == const.DATASET_RS:
                                # there is only background and cuticle in
                                # rs_dataset, ignore background
                                if label_names[sub_label] == 'cuticle':
                                    sub_images[img_label_name].append(
                                        img[x:x+rows, y:y+cols])
                                    sub_ids[img_label_name].append(_id)

            except Exception as e:
                logger.debug(f'Failed to open file {_id}.jpg: {e}')

        # convert to arrays
        sub_images_arr = []
        sub_labels_arr = []
        sub_ids_arr = []
        for k in sub_images:
            for sub_image in sub_images[k]:
                sub_images_arr.append(sub_image)
                if self.dataset_type == 'background':
                    sub_labels_arr.append(const.LABEL_MAP[k])
                elif self.dataset_type == 'rough_smooth':
                    sub_labels_arr.append(const.RS_LABEL_MAP[k])
            for sub_id in sub_ids[k]:
                sub_ids_arr.append(sub_id)

        assert len(sub_images_arr) == len(sub_labels_arr) == len(sub_ids_arr)

        sub_images_arr = np.array(sub_images_arr)
        sub_labels_arr = np.array(sub_labels_arr)
        sub_ids_arr = np.array(sub_ids_arr)

        if self.dataset_type == const.DATASET_BG:
            # convert cuticle_extra and antenna_base to cuticle
            idx = np.where(sub_labels_arr == const.LABEL_MAP['cuticle_extra'])
            sub_labels_arr[idx] = const.LABEL_MAP['cuticle']

            idx = np.where(sub_labels_arr == const.LABEL_MAP['antenna_base'])
            sub_labels_arr[idx] = const.LABEL_MAP['cuticle']

            idx = np.where(sub_labels_arr == const.LABEL_MAP['eye'])
            sub_labels_arr[idx] = const.LABEL_MAP['cuticle']

        # convert labels to be in range (0, 1)
        uniques = np.unique(sub_labels_arr)
        i = 0
        for unique in uniques:
            idx = np.where(sub_labels_arr == unique)
            sub_labels_arr[idx] = i
            i += 1

        assert len(sub_images_arr) == len(sub_labels_arr) == len(sub_ids_arr)

        # save subimage data and labels
        np.save(self.subimages_path, sub_images_arr)
        np.save(self.labels_path, sub_labels_arr)
        np.save(self.ids_path, sub_ids_arr)

        return sub_images_arr, sub_labels_arr, sub_ids_arr

    def _build_images(self, rebuild: bool):
        if rebuild:
            self.images, self.labels, self.ids = self.build_images()
            return

        try:
            self.images = np.load(self.images_path)
            self.labels = np.load(self.labels_path)
            self.ids = np.load(self.ids_path)
        except Exception as e:
            logger.debug(e)

            # build on failure-to-load
            self.images, self.labels, self.ids = self.build_images()

    def build_images(self):
        pass

    def get_label(self, _id: int) -> int:
        """
            Given an image id, return the label.
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

    def get_label_name(self, _id: int) -> str:
        return const.INT_RS_LABEL_MAP[self.get_label(_id)]

    def get_image(self, _id: int) -> np.ndarray:
        """
        Get image by ID.

        Args:
            _id (int): ID of the image.

        Returns:
            np.ndarray: Image as cv2 image object (numpy array).
        """
        path = f'./dataset/data/{_id}.jpg'
        img = cv2.imread(path)

        if img is None:
            logger.debug(f'Failed to open image {path}')

        return img

    def is_included(self, _id: int) -> bool:
        """
            Given an image id, return if the image is included from the dataset.
        """
        if _id in self.ids:
            return True
        return False

    def stratified_split(self, n: int):
        """
            [n]: number of samples per class
        """
        uniques = np.unique(self.labels, return_counts=True)

        # stratified sample n for each class in self.labels
        train_idxs = np.zeros((len(uniques[0])*n), dtype=np.int)
        for _class in range(len(uniques[0])):
            idx = np.where(self.labels == uniques[0][_class])[0]
            samples = self.rng.choice(idx, size=n, replace=False)
            train_idxs[_class*n:(_class+1)*n] = samples

        self.train_x = self.subimages[(train_idxs)]
        self.train_y = self.labels[(train_idxs)]

        test_idxs = np.array([idx for idx in range(
            len(self.labels)) if idx not in train_idxs])

        self.test_x = self.subimages[(test_idxs)]
        self.test_y = self.labels[(test_idxs)]

        assert len(self.train_y) + len(self.test_y) == len(self.labels)

        return self.train_x, self.train_y, self.test_x, self.test_y

    def get_ant_info(self, _id: int):
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
