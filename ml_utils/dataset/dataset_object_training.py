import torch
from torch.utils.data import Dataset

from ..transforms.bbox import apply_transform
from ..utils.utils import get_bbox_area, crop_out_of_shape, load_image


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

        image = load_image(f'{self.image_dir}/{image_id}')

        boxes = records[['x1', 'y1', 'x2', 'y2']].values
        boxes = crop_out_of_shape(boxes, image.shape)

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

        if self.transforms:
            target, image = apply_transform(self.transforms, target, image)

        return image, target, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]
