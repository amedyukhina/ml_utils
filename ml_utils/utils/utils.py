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


def load_image(fn, maxsize=None):
    image = io.imread(fn)
    if len(image.shape) < 3:
        image = np.array([image, image, image]).transpose(1, 2, 0)
    image = image.astype(np.float32)
    image /= image.max()

    if maxsize is not None:
        image = np.pad(image, [(0, maxsize - image.shape[0]), (0, maxsize - image.shape[1]), (0, 0)])

    return image


def remove_overlapping_boxes(boxes, scores, thr=0.1, return_full=False):
    iou = bops.box_iou(boxes, boxes).data.cpu().numpy()
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            if iou[i, j] > thr:
                ind = torch.tensor([i, j])[torch.argmin(torch.stack([scores[i], scores[j]]))]
                scores[ind] = 0
    if return_full:
        return boxes, scores
    else:
        return boxes[scores > 0]


def join_bboxes(*dfs, cl_name='class'):
    dfs = [wh_to_xy(df) for df in dfs]
    for i, df in enumerate(dfs):
        df[cl_name] = i
    return pd.concat(dfs, ignore_index=True)


def get_boxes_above_threshold(output, detection_thr):
    boxes = output['boxes']
    scores = output['scores']
    ind = scores > detection_thr
    boxes = boxes[ind]
    scores = scores[ind]
    return boxes, scores
