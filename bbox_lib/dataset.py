import numpy as np
import torch
from skimage import io
from torch.utils.data import Dataset


def crop_out_of_shape(boxes, shape):
    boxes[:, 0] = np.where(boxes[:, 0] < 0, 0, boxes[:, 0])
    boxes[:, 1] = np.where(boxes[:, 1] < 0, 0, boxes[:, 1])
    boxes[:, 2] = np.where(boxes[:, 2] >= shape[1], shape[1] - 1, boxes[:, 2])
    boxes[:, 3] = np.where(boxes[:, 3] >= shape[0], shape[0] - 1, boxes[:, 3])
    return boxes


def get_bbox_area(boxes):
    area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
    return area


def get_target(records, boxes, index):
    area = torch.as_tensor(get_bbox_area(boxes), dtype=torch.float32)
    # there is only one class
    labels = torch.ones((records.shape[0],), dtype=torch.int64)

    # suppose all instances are not crowd
    iscrowd = torch.zeros((records.shape[0],), dtype=torch.int64)

    target = dict(boxes=boxes,
                  labels=labels,
                  image_id=torch.tensor([index]),
                  area=area,
                  iscrowd=iscrowd)
    return target


def apply_transform(transforms, target, image):
    sample = {
        'image': image,
        'bboxes': target['boxes'],
        'labels': target['labels']
    }
    sample2 = transforms(**sample)
    while len(sample2['bboxes']) == 0:
        sample2 = transforms(**sample)

    image = sample2['image'].float()
    target['boxes'] = torch.stack(tuple(map(torch.tensor, zip(*sample2['bboxes'])))).permute(1, 0)
    target['boxes'] = target['boxes'].float()
    return target, image


class DatasetObjectTraining(Dataset):

    def __init__(self, dataframe, image_dir, transforms=None):
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms

    def __getitem__(self, index: int):

        image_id = self.image_ids[index]
        records = self.df[self.df['image_id'] == image_id]

        image = io.imread(f'{self.image_dir}/{image_id}')
        if len(image.shape) < 3:
            image = np.array([image, image, image]).transpose(1, 2, 0)
        image = image.astype(np.float32)
        image /= image.max()

        boxes = records[['x1', 'y1', 'x2', 'y2']].values
        boxes = crop_out_of_shape(boxes, image.shape)
        target = get_target(records, boxes, index)
        if self.transforms:
            target, image = apply_transform(self.transforms, target, image)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
