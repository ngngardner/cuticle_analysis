
import logging
import os

import cv2

from cuticle_analysis.analysis import image_analysis
from cuticle_analysis.dataset import Dataset
from cuticle_analysis.kviews import KViews

logging.basicConfig(filename='./logs/dataset.log', level=logging.INFO)
logger = logging.getLogger(__name__)


def k_views_experiments():
    # analytical variables
    random_seed = 5
    n_images = 5
    image_size = (512, 512)

    samples = {
        (9, 9): [80, 200, 250]
    }

    # build experiments based on data in [samples]
    experiments = []
    for size in samples.keys():
        case = 1
        for n in samples[size]:
            experiments.append({
                'subimage_size': size,
                'case': case,
                'n': n,
                'n_components': 3
            })
            experiments.append({
                'subimage_size': size,
                'case': case,
                'n': n,
                'n_components': 0
            })
            case += 1

    # run experiments
    i = 0
    for experiment in experiments:
        logger.info(f'Running experiment {i+1} out of {len(experiments)}')
        i += 1

        # load experiment info
        subimage_size = experiment['subimage_size']
        n = experiment['n']
        n_components = experiment['n_components']

        # load model
        data = Dataset(
            size=subimage_size,
            random_seed=random_seed)
        kviews = KViews(data)
        kviews.train(
            n=n,
            k=3,
            n_components=n_components)
        kviews.remap_centers()
        model = kviews

        # build output paths
        path = 'output/'
        path += f'{subimage_size[0]}_{subimage_size[1]}/'

        rough_path = f'{path}/rough_{experiment["case"]}'
        smooth_path = f'{path}/smooth_{experiment["case"]}'

        if n_components > 0:
            rough_path += '_pca'
            smooth_path += '_pca'

        rough_path += '.test.jpg'
        smooth_path += '.test.jpg'

        if not os.path.exists(path):
            os.makedirs(path)

        # run model on data and store results to the respective filepath
        rough_output = image_analysis(
            model, data, image_size=image_size, subimage_size=subimage_size,
            n_images=n_images, expected=1, random_seed=random_seed)
        cv2.imwrite(rough_path, rough_output)

        smooth_output = image_analysis(
            model, data, image_size=image_size, subimage_size=subimage_size,
            n_images=n_images, expected=2, random_seed=random_seed)
        cv2.imwrite(smooth_path, smooth_output)


if __name__ == '__main__':
    k_views_experiments()
