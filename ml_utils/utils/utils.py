import numpy as np


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
