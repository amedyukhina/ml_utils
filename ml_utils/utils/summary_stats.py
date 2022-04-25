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


def __get_dist_3d(centr, gt_centr):
    dist = np.zeros([len(gt_centr), len(centr)])
    for i in range(len(gt_centr)):
        for j in range(len(centr)):
            centers = np.array([center for center in [centr[j], gt_centr[i]]])
            dist[i, j] = np.sqrt(np.sum((centers[0] - centers[1]) ** 2))
    return dist


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


def accuracy_3d(centr, gt_centr, image_id, dist_thr):
    dist = __get_dist_3d(centr, gt_centr)
    inds = __get_match_indices(dist, dist_thr)
    if len(inds) > 0:
        tp = len(dist[inds])
        dist_err = np.mean(dist[inds])
    else:
        tp = dist_err = 0

    stats = {'image_id': image_id, 'n ground truth': len(gt_centr), 'n detected': len(centr),
             'false negatives': len(gt_centr) - tp, 'false positives': len(centr) - tp,
             'true positives': tp, 'distance error pix': dist_err}
    return pd.Series(stats).to_frame().transpose()


def summarize_accuracy(df):
    df = pd.concat(df, ignore_index=True)
    tp = np.sum(df['true positives'])
    n_detected = np.sum(df['n detected'])
    n_gt = np.sum(df['n ground truth'])
    recall = tp / n_gt
    if n_detected > 0:
        prec = tp / n_detected
    else:
        prec = np.nan
    if recall > 0:
        fscore = 2 * recall * prec / (prec + recall)
    else:
        fscore = np.nan
    if tp > 0:
        dist_err = np.sum(df['distance error pix'] * df['true positives']) / tp
    else:
        dist_err = np.nan
    stats = {'Recall': recall, 'Precision': prec, 'F Score': fscore,
             'Distance error pix': dist_err}
    if 'Jaccard index' in df.columns:
        if tp > 0:
            stats['Jaccard index'] = np.sum(df['Jaccard index'] * df['true positives']) / tp
        else:
            stats['Jaccard index'] = np.nan

    return stats
