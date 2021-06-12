
import argparse
import os

import cv2

from cuticle_analysis.analysis import image_analysis
from cuticle_analysis.dataset import Dataset
from cuticle_analysis.e2e import EndToEnd
from cuticle_analysis.cnn import CNN
from cuticle_analysis.kviews import KViews

if not os.path.exists('./output'):
    os.makedirs('./output')

parser = argparse.ArgumentParser(description='Analyze ant cuticles.')
parser.add_argument('--model', type=str, help='model to use')
parser.add_argument('--train', action='store_true',
                    help='train a new model. otherwise, load weights')

args = parser.parse_args()


if __name__ == '__main__':
    # experimental variables
    subimage_size = (32, 32)    # size of subimages to use

    # kviews
    n = 250                     # number of samples per class
    k = 3                       # number of classes
    n_components = 3            # number of pca components

    # cnn
    test_size = 0.2             # percent test data
    epochs = 10                 # training epochs

    # analytical variables
    random_seed = 5             # use random seed for consistent results
    n_images = 5                # number of images to analyze
    image_size = (512, 512)     # output image size

    bg_data = Dataset(
        size=subimage_size,
        dataset_type='background',
        random_seed=random_seed,
        rebuild=True)
    rs_data = Dataset(
        size=subimage_size,
        dataset_type='rough_smooth',
        random_seed=random_seed,
        rebuild=True)

    # cnn model for classifying background
    cnn = CNN(bg_data)
    if args.train:
        cnn.train(epochs=epochs, test_size=test_size)
        cnn.save_weights()
    else:
        cnn.load_weights()

    # kviews model for rough/smooth
    kviews = KViews(rs_data)
    kviews.train(n=n, k=k, n_components=n_components)
    kviews.remap_centers()

    model = EndToEnd(cnn, kviews)

    # Rough Analysis
    rough_output = image_analysis(
        model, rs_data, image_size=image_size, subimage_size=subimage_size,
        n_images=n_images, expected=1, random_seed=random_seed)
    cv2.imwrite('output/rough.test.jpg', rough_output)

    # Smooth Analysis
    smooth_output = image_analysis(
        model, rs_data, image_size=image_size, subimage_size=subimage_size,
        n_images=n_images, expected=2, random_seed=random_seed)
    cv2.imwrite('output/smooth.test.jpg', smooth_output)

    # Cluster Analysis
    if model.name == 'k-views':
        model.analyze()
