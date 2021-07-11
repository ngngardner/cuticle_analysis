
import re
import glob
import json
import logging
from typing import Tuple, List

import cv2
import numpy as np

from .dataset import Dataset
from .utils import get_label_names
from .. import const

logger = logging.getLogger(__name__)


class SubDataset(Dataset):
    def __init__(self,
                 size: tuple,
                 name: str,
                 d_type: str = None,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        """Base class for building subimage datasets. Overrides
        Dataset.build_images to cut subimages of $size from the original images.
        """
        name = f'{name}_sub'
        super().__init__(size,
                         name=name,
                         d_type=d_type,
                         excludes=excludes,
                         random_seed=random_seed,
                         rebuild=rebuild,
                         save=save)

    def preprocess(self, img: np.ndarray) -> List[np.ndarray]:
        """Apply preprocessing step to the image

        Args:
            img (np.ndarray): Orginal image.

        Returns:
            np.ndarray: Updated image with preprocessing, using a list to allow
            for multiple images to be output by preprocessing.
        """
        return [img]

    def build_images(self, save: bool) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Builds the subimage dataset.

        Args:
            save (bool): Save the generated files.

        Returns:
            subimages (np.ndarray): Array of generated subimages.
            labels (np.ndarray): Ground-truth label of each subimage.
            ids (np.ndarray): Original sample id of each subimage.
        """
        rows = self.size[0]
        cols = self.size[1]

        logger.info(
            f'Generating dataset of size {rows} by {cols}.')

        # sub image data and original ids
        sub_images = {}
        sub_ids = {}

        # if using rs or bg dataset, only use images with json
        files = glob.glob(
            f'./dataset/{const.DS_MAP[self.d_type]}/*.json')

        logger.info(f'Using {len(files)} images.')

        for file in files:
            # load image data based on avaiable jsons
            _id = int(re.findall('[0-9]+', file)[0])

            try:
                with open(file) as f:
                    data = json.load(f)

                img = self.get_image(_id)
                img_label_name = self.get_label_name(_id)
                lbl, label_names = get_label_names(img, data)

                # preprocessing returns a list of images to use
                p_imgs = self.preprocess(img)

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

                            for p_img in p_imgs:
                                if self.d_type == const.DATASET_BG:
                                    # only add background if there are less bg samples
                                    # than cuticle
                                    if label_names[sub_label] == '_background_':
                                        if len(sub_images['_background_']) \
                                                < len(sub_images['cuticle']):
                                            sub_images[label_names[sub_label]].append(
                                                p_img[x:x+rows, y:y+cols])
                                            sub_ids[label_names[sub_label]].append(
                                                _id)
                                    else:
                                        sub_images[label_names[sub_label]].append(
                                            p_img[x:x+rows, y:y+cols])
                                        sub_ids[label_names[sub_label]].append(
                                            _id)

                                elif (self.d_type == const.DATASET_RS
                                        and label_names[sub_label] == 'cuticle'):
                                    # there is only background and cuticle in
                                    # rs_dataset, ignore background
                                    sub_images[img_label_name].append(
                                        p_img[x:x+rows, y:y+cols])
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
                if self.d_type == const.DATASET_BG:
                    sub_labels_arr.append(const.LABEL_MAP[k])
                elif self.d_type == const.DATASET_RS:
                    sub_labels_arr.append(const.RS_LABEL_MAP[k])
            for sub_id in sub_ids[k]:
                sub_ids_arr.append(sub_id)

        assert len(sub_images_arr) == len(sub_labels_arr) == len(sub_ids_arr)

        sub_images_arr = np.array(sub_images_arr)
        sub_labels_arr = np.array(sub_labels_arr)
        sub_ids_arr = np.array(sub_ids_arr)

        if self.d_type == const.DATASET_BG:
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
        if save:
            np.save(self.images_path, sub_images_arr)
            np.save(self.labels_path, sub_labels_arr)
            np.save(self.ids_path, sub_ids_arr)

        return sub_images_arr, sub_labels_arr, sub_ids_arr
