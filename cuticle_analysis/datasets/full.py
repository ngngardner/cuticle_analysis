
import re
import glob
import logging
from typing import Tuple, List

import cv2
import numpy as np

from .dataset import Dataset

logger = logging.getLogger(__name__)


class FullDataset(Dataset):
    def __init__(self,
                 size: tuple,
                 name: str,
                 d_type: str = None,
                 excludes: list = None,
                 random_seed: int = None,
                 rebuild: bool = False,
                 save: bool = False):
        """Base class for building full-size datasets. Overrides
        Dataset.build_images to resize images to $size
        """
        name = f'{name}_full'
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
        """Builds the image dataset.

        Args:
            save (bool): Save the generated files.

        Returns:
            subimages (np.ndarray): Array of generated images.
            labels (np.ndarray): Ground-truth label of each image.
            ids (np.ndarray): Original sample id of each image.
        """
        rows = self.size[0]
        cols = self.size[1]

        logger.info(
            f'Generating dataset of size {self.size[0]} by {self.size[1]}.')

        files = glob.glob(f'./dataset/data/*.jpg')

        images = {}
        ids = {}

        for file in files:
            try:
                _id = int(re.findall('[0-9]+', file)[0])
                img = self.get_image(_id)
                img_label = self.get_label(_id)
                if img_label not in images.keys():
                    images[img_label] = []
                    ids[img_label] = []

                p_imgs = self.preprocess(img)
                for p_img in p_imgs:
                    images[img_label].append(cv2.resize(p_img, (rows, cols)))
                    ids[img_label].append(_id)

            except Exception as e:
                logger.debug(f'Failed to open file {_id}.jpg: {e}')

        images_arr = []
        labels_arr = []
        ids_arr = []
        for k in images:
            for image in images[k]:
                images_arr.append(image)
                labels_arr.append(k)
            for _id in ids[k]:
                ids_arr.append(_id)

        assert len(images_arr) == len(labels_arr) == len(ids_arr)

        images_arr = np.array(images_arr)
        labels_arr = np.array(labels_arr)
        ids_arr = np.array(ids_arr)

        # save image data and labels
        if save:
            np.save(self.images_path, images_arr)
            np.save(self.labels_path, labels_arr)
            np.save(self.ids_path, ids_arr)

        return images_arr, labels_arr, ids_arr
