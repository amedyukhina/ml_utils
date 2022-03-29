import numpy as np
import pandas as pd
import torch
import torchvision.ops.boxes as bops


class Averager:
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0


def __get_scores(bboxes, gt_boxes):
    jaccard = np.zeros([len(gt_boxes), len(bboxes)])
    dist = np.zeros([len(gt_boxes), len(bboxes)])
    for i in range(len(gt_boxes)):
        for j in range(len(bboxes)):
            jaccard[i, j] = bops.box_iou(torch.tensor([gt_boxes[i]], dtype=torch.float),
                                         torch.tensor([bboxes[j]], dtype=torch.float)).data.cpu().numpy()[0, 0]
            centers = np.array([[(box[0] + box[2]) / 2, (box[1] + box[3]) / 2] for box in [bboxes[j], gt_boxes[i]]])
            dist[i, j] = np.sqrt(np.sum((centers[0] - centers[1]) ** 2))
    return jaccard, dist


def __get_match_indices(dist, dist_thr):
    dist2 = dist.copy()
    maxval = 10 ** 10
    inds = []

    for i in range(min(len(dist), len(dist[0]))):
        ind = np.unravel_index(np.argmin(dist2, axis=None), dist2.shape)
        if dist[ind] <= dist_thr:
            inds.append(ind)
        else:
            break
        dist2[ind[0], :] = maxval
        dist2[:, ind[1]] = maxval
    return tuple(np.array(inds).transpose())


def accuracy(bboxes, gt_boxes, image_id, dist_thr):
    jaccard, dist = __get_scores(bboxes, gt_boxes)
    inds = __get_match_indices(dist, dist_thr)
    if len(inds) > 0:
        tp = len(dist[inds])
        dist_err = np.mean(dist[inds])
        jc = np.mean(jaccard[inds])
    else:
        tp = dist_err = jc = 0

    stats = {'image_id': image_id, 'n ground truth': len(gt_boxes), 'n detected': len(bboxes),
             'false negatives': len(gt_boxes) - tp, 'false positives': len(bboxes) - tp,
             'true positives': tp, 'distance error pix': dist_err,
             'Jaccard index': jc}
    return pd.Series(stats).to_frame().transpose()
