import numpy as np

from ..predict.predict_bbox import detect_bboxes
from ..utils.convert import wh_to_xy
from ..utils.summary_stats import accuracy, summarize_accuracy


def evaluate_accuracy(df_gt, model_fn, input_dir, config):
    df_gt = wh_to_xy(df_gt)

    df = detect_bboxes(input_dir=input_dir,
                       model_fn=model_fn,
                       batch_size=config.batch_size,
                       overlap_threshold=config.overlap_thr,
                       detection_threshold=config.detection_thr)

    cols = ['x1', 'y1', 'x2', 'y2']
    accuracy_df = []
    for image_id in df_gt['image_id'].unique():
        bboxes = df[df['image_id'] == image_id][cols].values
        gt_boxes = np.array(df_gt[df_gt['image_id'] == image_id][cols].values)
        accuracy_df.append(accuracy(bboxes, gt_boxes, image_id=image_id, dist_thr=config.dist_thr))
    return summarize_accuracy(accuracy_df)
