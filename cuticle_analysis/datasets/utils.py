
import math
import uuid
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import PIL.Image


def shape_to_mask(img_shape, points, shape_type=None, line_width=10, point_size=5):
    mask = np.zeros(img_shape[:2], dtype=np.uint8)
    mask = PIL.Image.fromarray(mask)
    draw = PIL.ImageDraw.Draw(mask)
    xy = [tuple(point) for point in points]
    if shape_type == "circle":
        assert len(xy) == 2, "Shape of shape_type=circle must have 2 points"
        (cx, cy), (px, py) = xy
        d = math.sqrt((cx - px) ** 2 + (cy - py) ** 2)
        draw.ellipse([cx - d, cy - d, cx + d, cy + d], outline=1, fill=1)
    elif shape_type == "rectangle":
        assert len(xy) == 2, "Shape of shape_type=rectangle must have 2 points"
        draw.rectangle(xy, outline=1, fill=1)
    elif shape_type == "line":
        assert len(xy) == 2, "Shape of shape_type=line must have 2 points"
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "linestrip":
        draw.line(xy=xy, fill=1, width=line_width)
    elif shape_type == "point":
        assert len(xy) == 1, "Shape of shape_type=point must have 1 points"
        cx, cy = xy[0]
        r = point_size
        draw.ellipse([cx - r, cy - r, cx + r, cy + r], outline=1, fill=1)
    else:
        assert len(xy) > 2, "Polygon must have points more than 2"
        draw.polygon(xy=xy, outline=1, fill=1)
    mask = np.array(mask, dtype=bool)
    return mask


def shapes_to_label(img_shape, shapes, label_name_to_value):
    cls = np.zeros(img_shape[:2], dtype=np.int32)
    ins = np.zeros_like(cls)
    instances = []
    for shape in shapes:
        points = shape["points"]
        label = shape["label"]
        group_id = shape.get("group_id")
        if group_id is None:
            group_id = uuid.uuid1()
        shape_type = shape.get("shape_type", None)

        cls_name = label
        instance = (cls_name, group_id)

        if instance not in instances:
            instances.append(instance)
        ins_id = instances.index(instance) + 1
        cls_id = label_name_to_value[cls_name]

        mask = shape_to_mask(img_shape[:2], points, shape_type)
        cls[mask] = cls_id
        ins[mask] = ins_id

    return cls, ins


def convert_labels(label: pd.Series) -> pd.Series:
    """Convert labels to numbers

    Args:
        label (pd.Series): [description]

    Returns:
        pd.Series: [description]
    """
    raise NotImplementedError


def convert_labels_rs(label: pd.Series) -> pd.Series:
    """Convert labels to rough (0) and smooth (1)

    Args:
        label (pd.Series): [description]

    Returns:
        pd.Series: [description]
    """
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
    lbl, _ = shapes_to_label(
        img.shape, data["shapes"], label_name_to_value
    )

    label_names = [None] * (max(label_name_to_value.values()) + 1)
    for name, value in label_name_to_value.items():
        label_names[value] = name

    return lbl, label_names
