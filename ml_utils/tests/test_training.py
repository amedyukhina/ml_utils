import os
import shutil
import unittest

import numpy as np
from ddt import ddt
from skimage import io

from ..model.faster_rcnn import load_model_for_training
from ..train.dataloader_bbox import get_data_loaders
from ..train.train_bbox import train
from ..utils.visualize import draw_bbox

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

    def test_training(self):
        tr_dl, val_dl = get_data_loaders(os.path.join(INPUT_DIR, 'bboxes.csv'),
                                         input_dir=os.path.join(INPUT_DIR, 'img'),
                                         val_fraction=0.2, batch_size=2, num_workers=2)
        model = load_model_for_training()
        train(model, tr_dl, val_dl, num_epochs=2, lr=0.01, momentum=0.9, weight_decay=0.0005,
              step_size=3, gamma=0.1, output_dir=INPUT_DIR + '/../tmp/model')
        self.assertTrue(os.path.exists(INPUT_DIR + '/../tmp/model/weights_best.pth'))
        shutil.rmtree(INPUT_DIR + '/../tmp')


if __name__ == '__main__':
    unittest.main()
