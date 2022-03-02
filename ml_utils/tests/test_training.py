import os
import shutil
import time
import unittest

import numpy as np
import pandas as pd
import wandb
from ddt import ddt, data
from skimage import io

from ..model.faster_rcnn import load_model_for_training
from ..predict.predict_bbox import detect_bboxes
from ..train.dataloader_bbox import get_data_loaders
from ..train.train_bbox import train
from ..utils.utils import join_bboxes
from ..utils.visualize import draw_bbox, overlay_bboxes_batch

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data'


@ddt
class TestTraining(unittest.TestCase):

    def test_input_data(self):
        tr_dl, val_dl = get_data_loaders(os.path.join(INPUT_DIR, 'bboxes.csv'),
                                         input_dir=os.path.join(INPUT_DIR, 'img'),
                                         val_fraction=0.2, batch_size=2, num_workers=2)

        for images, targets, image_ids in tr_dl:
            for i in range(len(images)):
                boxes = targets[i]['boxes'].cpu().numpy().astype(np.int32)
                sample = images[i].permute(1, 2, 0).cpu().numpy()

                for box in boxes:
                    sample = draw_bbox(sample, [box[1], box[0], box[3], box[2]], color=(1, 0, 0))

                io.imshow(sample)
                # import pylab as plt
                # plt.show()

    @data(
        False
    )
    def test_training(self, log_progress):
        tr_dl, val_dl = get_data_loaders(os.path.join(INPUT_DIR, 'bboxes.csv'),
                                         input_dir=os.path.join(INPUT_DIR, 'img'),
                                         val_fraction=0.2, batch_size=2, num_workers=2)
        model = load_model_for_training()
        config = dict(num_epochs=5, lr=0.01, momentum=0.9, weight_decay=0.0005,
                      step_size=3, gamma=0.1)
        if log_progress is False:
            os.environ['WANDB_MODE'] = 'offline'
        wandb.init(project='test_project', config=config)
        if log_progress:
            model_name = wandb.run.name
        else:
            model_name = str(time.time())
        train(model, tr_dl, val_dl, config=config, log_progress=log_progress,
              model_dir=INPUT_DIR + '/../tmp/model', model_name=model_name)
        wandb.finish()

        df = detect_bboxes(input_dir=os.path.join(INPUT_DIR, 'img'),
                           model_fn=INPUT_DIR + rf'/../tmp/model/{model_name}/weights_best.pth',
                           batch_size=2, overlap_threshold=0.1, detection_threshold=0.1)
        self.assertGreater(len(df), 0)

        df_gt = pd.read_csv(os.path.join(INPUT_DIR, 'bboxes.csv'))
        df = join_bboxes(df_gt, df, cl_name='class')

        overlay_bboxes_batch(df=df, input_dir=os.path.join(INPUT_DIR, 'img'),
                             output_dir=INPUT_DIR + rf'/../tmp/overlay/{model_name}',
                             palette=[(0, 255, 0), (255, 0, 0)])

        self.assertTrue(os.path.exists(INPUT_DIR +
                                       rf'/../tmp/model/{model_name}/weights_best.pth'))
        self.assertEqual(len(os.listdir(INPUT_DIR + rf'/../tmp/overlay/{model_name}')), 5)

        shutil.rmtree(INPUT_DIR + '/../tmp')


if __name__ == '__main__':
    unittest.main()
