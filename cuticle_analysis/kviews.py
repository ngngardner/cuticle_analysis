
import logging

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import rand_score, adjusted_rand_score, cluster

from .dataset import Dataset

logger = logging.getLogger(__name__)


class KViews():
    def __init__(self, data: Dataset):
        self.data = data
        self.name = 'k-views'

    def metadata(self):
        return [
            f'Model: {self.name}, k: {self.k}, n_iter: {self.model.n_iter_}',
            f'n: {self.n}, N: {self.n*self.k}',
            f'n_components: {self.n_components}',
        ]

    def train(self, n: int, k: int, n_components: int):
        """
            Train k-views model based on dataset.

            [n]: number of samples per class
            [k]: number of classes
            [n_components]: number of pca components to use
        """
        self.n = n
        self.k = k
        self.n_components = n_components
        self.model = KMeans(n_clusters=k)

        # test/train split
        train_x, train_y, test_x, test_y = self.data.stratified_split(n)
        train_x = train_x.reshape(train_x.shape[0], -1)

        # fit pca
        if n_components > 0:
            self.pca = PCA(n_components=n_components).fit(train_x)
            train_x = self.pca.transform(train_x)

        # train kmeans
        self.model.fit(train_x)

    def predict_test(self):
        test_x = self.data.test_x
        test_x = test_x.reshape(test_x.shape[0], -1)
        self.pred = self._predict(test_x, reshape=False)
        self.expected = self.data.test_y

    def remap_centers(self):
        """
            Using ground truth, change the cluster ids to the expected.
        """
        try:
            self.pred
            self.expected
        except Exception as e:
            logger.debug(e)

            # build on failure-to-load
            self.predict_test()

        c = cluster.contingency_matrix(
            self.pred, self.expected, eps=1, dtype=int)
        self.adjusted_centers = linear_sum_assignment(c, maximize=True)
        self.remapped = True

        logger.info(self.adjusted_centers)

    def analyze(self):
        logger.info(f'Analyzing k-views model...')

        test_x = self.data.test_x
        test_x = test_x.reshape(test_x.shape[0], -1)
        self.pred = self.predict(test_x)
        self.expected = self.data.test_y

        logger.info(f'Predicted:')
        uniques = np.unique(self.pred, return_counts=True)
        for _class in range(len(uniques[0])):
            logger.info(f'\t{uniques[0][_class]}: {uniques[1][_class]}')

        logger.info(f'Expected:')
        uniques = np.unique(self.expected, return_counts=True)
        for _class in range(len(uniques[0])):
            logger.info(f'\t{uniques[0][_class]}: {uniques[1][_class]}')

        logger.info(f'Rand Score: {rand_score(self.expected, self.pred)}')
        logger.info(
            f'Adjusted Rand Score: {adjusted_rand_score(self.expected, self.pred)}')

    def _predict(self, images: np.ndarray, reshape=True) -> np.ndarray:
        """
        Helper function for "predict".

        Args:
            images (np.ndarray): 4-d array of images (sample, length, width, channels)
            reshape (bool, optional): Flatten images if true. Defaults to True.

        Returns:
            [type]: [description]
        """
        if reshape:
            images = images.reshape(images.shape[0], -1)
        # if pca components > 0, then transform
        if self.n_components > 0:
            images = self.pca.transform(images)
        return self.model.predict(images)

    def predict(self, images: np.ndarray):
        preds = self._predict(images)
        if self.remapped:
            preds = np.array(
                [self.adjusted_centers[1][pred] for pred in preds])
        return preds
