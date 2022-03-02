import numpy as np
import pandas as pd
import torch
import torchvision.ops.boxes as bops
from skimage import io

from .convert import wh_to_xy


def collate_fn(batch):
    return tuple(zip(*batch))


def crop_out_of_shape(boxes, shape):
    boxes[:, 0] = np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
    boxes[:, 1] = np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
    boxes[:, 2] = np.where(boxes[:, 2] >= shape[1], shape[1] - 1, boxes[:, 2])
    boxes[:, 3] = np.where(boxes[:, 3] >= shape[0], shape[0] - 1, boxes[:, 3])
    return boxes


def get_bbox_area(boxes):
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    return area


def load_image(fn):
    image = io.imread(fn)
    if len(image.shape) < 3:
        image = np.array([image, image, image]).transpose(1, 2, 0)
    image = image.astype(np.float32)
    image /= image.max()
    return image


def remove_overlapping_boxes(pred, thr=0.1, col='image_id'):
    """
    Identify bounding boxes that overlap with a IOU score higher than `thr` and
    remove the box with a lower confidence score.

    Parameters
    ----------
    pred : pd.DataFrame
        Dataframe with predicted bounding box coordinates.
    thr : float, optional
        Maximum allowed intersection-over-union (IOU) score for two bounding boxes.
        If two boxes overlap with a higher score, the box with a lower confidence score will be removed
    col : str, optional
        Column name to specify image ID.
        Default is 'image_id'

    Returns
    -------
    pd.DataFrame
        Dataframe with overlapping boxes removed.
    """
    for image_id in pred[col].unique():
        cp = pred[pred[col] == image_id]
        if len(cp) > 2:
            boxes = cp[['x1', 'y1', 'x2', 'y2']].values
            scores = np.array(cp['scores'])
            for i in range(len(boxes)):
                for j in range(i + 1, len(boxes)):
                    box1 = torch.tensor(np.array([boxes[i]]), dtype=torch.float)
                    box2 = torch.tensor(np.array([boxes[j]]), dtype=torch.float)
                    iou = bops.box_iou(box1, box2).data.cpu().numpy()[0, 0]
                    if iou > thr:
                        ind = np.array([i, j])[np.argmin([scores[i], scores[j]])]
                        scores[ind] = 0
            pred.loc[cp.index, 'scores'] = scores
    pred = pred[pred['scores'] > 0].reset_index(drop=True)
    return pred


def join_bboxes(*dfs, cl_name='class'):
    dfs = [wh_to_xy(df) for df in dfs]
    for i, df in enumerate(dfs):
        df[cl_name] = i
    return pd.concat(dfs, ignore_index=True)
