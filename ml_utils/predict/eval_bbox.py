import numpy as np
import pandas as pd

from ..utils.convert import wh_to_xy
from ..utils.summary_stats import accuracy, summarize_accuracy


def evaluate_accuracy(df, df_gt, dist_thr, return_full=False):
    df_gt = wh_to_xy(df_gt)

    cols = ['x1', 'y1', 'x2', 'y2']
    accuracy_df = []
    for image_id in df_gt['image_id'].unique():
        bboxes = df[df['image_id'] == image_id][cols].values
        gt_boxes = np.array(df_gt[df_gt['image_id'] == image_id][cols].values)
        accuracy_df.append(accuracy(bboxes, gt_boxes, image_id=image_id, dist_thr=dist_thr))
    stats = summarize_accuracy(accuracy_df)
    if return_full:
        return stats, pd.concat(accuracy_df, ignore_index=True)
    else:
        return stats
