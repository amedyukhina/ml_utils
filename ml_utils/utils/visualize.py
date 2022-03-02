import os

import numpy as np
import pandas as pd
from am_utils.utils import imsave, walk_dir
from joblib import Parallel, delayed
from skimage import draw
from skimage import io
from tqdm import tqdm

from .convert import wh_to_xy


def overlay_bboxes_batch(df, input_dir, n_jobs=20, **kwargs):
    image_ids = [fn[len(input_dir.rstrip('/')) + 1:] for fn in walk_dir(input_dir)]

    def __overlay_bboxes(fn, dfr, inp_dir, output_dir, **kwargs):
        img = io.imread(os.path.join(inp_dir, fn))
        img = overlay_bboxes(img, dfr[dfr['image_id'] == fn], **kwargs)
        imsave(os.path.join(output_dir, fn.replace('/', '_')), img)

    Parallel(n_jobs=n_jobs)(delayed(__overlay_bboxes)(
        fn, df, inp_dir=input_dir, **kwargs
    ) for fn in tqdm(image_ids))


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
    if len(df) > 0:
        df = wh_to_xy(df)
        if img.shape[-1] != 3:
            img = np.array([img, img, img]).transpose(1, 2, 0)
        if 'class' in df.columns:
            cl = np.array(df['class'])
        else:
            cl = [0] * len(df)

        for i in range(len(df)):
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
