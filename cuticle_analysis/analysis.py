
import logging

import cv2
import numpy as np
from numpy.random import default_rng

from .datasets import Dataset
from . import const

logger = logging.getLogger(__name__)

colors = [
    (244, 236, 214),
    (136, 183, 181),
    (49, 10, 49),
    (110, 125, 171),
    (87, 98, 213)]


def draw_texts(img,
               texts: list,
               initial: tuple,
               size: float = 0.5,
               from_bottom: bool = False,
               color: tuple = (255, 255, 255)):
    """
        [size]: font size
    """
    font = cv2.FONT_HERSHEY_COMPLEX
    line_type = 2

    line = 1
    line_height = 24
    for text in texts:
        if from_bottom:
            # bottom-left corner of text in bottom-right
            org = (initial[0], initial[1]-(line*line_height))
        else:
            # bottom-left corner of text in top-right
            org = (initial[0], initial[1]+(line*line_height))

        # write twice for outline
        cv2.putText(
            img, text, org,
            font, size,
            (0, 0, 0),
            lineType=line_type,
            thickness=5)
        cv2.putText(
            img, text, org,
            font, size,
            color,
            lineType=line_type,
            thickness=2)

        line += 1


def highlight(model, img: np.ndarray, size: tuple):
    assert len(size) == 2
    rows = size[0]
    cols = size[1]

    highlighted = np.zeros(img.shape)
    subimages = []
    for x in range(0, img.shape[0]-rows, rows):
        for y in range(0, img.shape[1]-cols, cols):
            subimage = img[x:x+rows, y:y+cols]
            subimages.append(subimage)

    preds = model.predict(np.array(subimages))
    idx = 0
    for x in range(0, img.shape[0]-rows, rows):
        for y in range(0, img.shape[1]-cols, cols):
            color = colors[preds[idx]]
            highlighted[x:x+rows, y:y+cols, :] = color
            idx += 1

    # TODO: highlight areas that were missed with colors[0]
    return highlighted, preds


def build_legend(height, width, n_classes):
    """
        Build horizontal legend
    """
    legend = np.zeros((height, width, 3))
    class_width = int(width/n_classes)

    for i in range(n_classes):
        text = const.INT_RS_LABEL_MAP[i]
        legend[:, class_width*i:class_width*(i+1), :] = colors[i]
        draw_texts(legend, [text],
                   (12+(class_width*i), 16), size=1)

    return legend


def mapped_image(img, pred, legend, image_size, legend_height):
    """
        Return original image with higlighted image to the right.
        Includes legend.
    """
    rows = image_size[0]
    cols = image_size[1]

    res = np.zeros((rows+legend_height, cols*2, 3))

    # input image on left
    res[legend_height:rows+legend_height,
        :cols, :] = cv2.resize(img, image_size)

    # output image on right
    res[legend_height:rows+legend_height,
        cols:, :] = cv2.resize(pred, image_size)

    # legend
    res[:legend_height, :, :] = legend

    return res


def image_analysis(model,
                   data: Dataset,
                   image_size: tuple,
                   subimage_size: tuple,
                   n_images: int,
                   expected: int,
                   random_seed: int = None):
    """
        [expected]: label id to analyze
    """
    logger.info(f'Analyzing {n_images} images.')
    rng = default_rng(random_seed)

    # for creating a legend in each image
    legend_height = 64
    subheight = image_size[0]+legend_height
    legend = build_legend(
        legend_height, width=image_size[1]*2, n_classes=3)

    # store images that were already selected
    images = []

    # output array to hold result
    output = np.zeros((
        (subheight)*n_images,
        image_size[1] * 2,
        3
    ))

    while True:
        # get a random image and its id
        _id = rng.choice(data.img_meta[0])
        label = data.get_label(_id)

        # if the image wasn't included as training data, and is
        # of the type we are expecting, reserve to first 300 for analysis
        if not data.is_included(_id) and _id not in images \
                and label == expected and _id < 300:
            img = data.get_image(_id)
            if img is not None:
                logger.info(
                    f'Analyzing image {_id}: {len(images)+1}/{n_images}.')

                # highlight image based on prediction label
                highlighted, preds = highlight(model, img, subimage_size)

                # side-by-side image
                m_img = mapped_image(
                    img, highlighted, legend, image_size, legend_height)

                # write some text about the image and model used
                texts = [
                    f'Label: {data.get_label_name(_id)}, '
                    + f'ID: {_id}, Size: {subimage_size}',
                ]
                for metadata in model.metadata():
                    texts.append(metadata)

                draw_texts(m_img, texts, (12, legend_height))

                # draw prediction text

                # where to start placing prediction info
                pred_height = legend_height+(24*len(texts))

                try:
                    pred_percent = np.unique(preds, return_counts=True)
                    rough_count = pred_percent[1][1]
                    smooth_count = pred_percent[1][2]

                    rough_percent = rough_count/(rough_count+smooth_count)
                    smooth_percent = smooth_count/(rough_count+smooth_count)

                    draw_texts(m_img, ['Prediction:'], (12, pred_height))

                    rough_percent_str = f'{np.round(rough_percent, 2)}% Rough'
                    smooth_percent_str = f'{np.round(smooth_percent, 2)}% Smooth'

                    if (rough_percent > smooth_percent):
                        if label == 1:  # rough
                            # print green rough
                            draw_texts(m_img, [rough_percent_str],
                                       (12, pred_height+24),
                                       color=(0, 255, 0))
                        else:
                            # print red rough
                            draw_texts(m_img, [rough_percent_str],
                                       (12, pred_height+24),
                                       color=(0, 0, 255))

                        # print smooth percent
                        draw_texts(m_img, [smooth_percent_str],
                                   (12, pred_height+48))
                    elif (smooth_percent > rough_percent):
                        # print rough percent
                        draw_texts(m_img, [rough_percent_str],
                                   (12, pred_height+24))

                        if label == 1:  # rough
                            # print red smooth
                            draw_texts(m_img, [smooth_percent_str],
                                       (12, pred_height+48),
                                       color=(0, 0, 255))
                        else:
                            # print green smooth
                            draw_texts(m_img, [smooth_percent_str],
                                       (12, pred_height+48),
                                       color=(0, 255, 0))

                except Exception as e:
                    logger.debug(f'Failed to load prediction percent: {e}')

                # write some text about the ant data
                ant_info = data.get_ant_info(_id)
                draw_texts(
                    m_img, ant_info,
                    (12, image_size[0]+legend_height),
                    from_bottom=True)

                output[
                    subheight*len(images):subheight*(len(images)+1),
                    :,
                    :,
                ] = m_img

                images.append(_id)

                if len(images) >= n_images:
                    return output
