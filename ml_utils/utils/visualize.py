import numpy as np
import pandas as pd
from skimage import draw


def overlay_bboxes(img, df, palette=None, color=None):
    """
    Overlay bounding boxes on the input image.

    Parameters
    ----------
    img : np.ndarray
        Input image.
    df : pd.DataFrame
        Bounding box coordinates as pandas dataframe.
    palette : list of tuples, optional
        Color palette to apply to bounding boxes of different classes.
        Default is None.
    color : tuple, optional
        Color of the bounding boxes.
        If None, red color is applied: (255, 0, 0).
        Default is None.

    Returns
    -------
    img : np.ndarray
        Input image with overlaid bounding boxes.

    """
    if img.shape[-1] != 3:
        img = np.array([img, img, img]).transpose(1, 2, 0)
    if 'class' in df.columns:
        cl = np.array(df['class'])
    else:
        cl = [0] * len(df)

    for i in range(len(df)):
        if 'height' in df.columns and df['height'].iloc[i] > -1:
            box = df.iloc[i][['y', 'x', 'height', 'width']].values
            box[2] = box[2] + box[0]
            box[3] = box[3] + box[1]
        else:
            box = df.iloc[i][['y1', 'x1', 'y2', 'x2']].values

        if palette is not None and cl[i] < len(palette):
            color = palette[cl[i]]
        elif color is None:
            color = (255, 0, 0)

        img = draw_bbox(img, box, color)
    return img


def draw_bbox(img, box, color=(255, 255, 255)):
    rr, cc = np.int_(draw.rectangle_perimeter((box[0], box[1]),
                                              (box[2], box[3])))
    ind = np.where((rr >= 0) & (rr < img.shape[0]) & (cc >= 0) & (cc < img.shape[1]))
    img[rr[ind], cc[ind]] = color
    return img
