import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2


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


def get_train_transform():
    return A.Compose([
        A.Flip(p=0.5),
        A.Rotate(p=1),
        A.Transpose(p=0.5),
        A.RandomBrightness(p=0.5),
        A.RandomContrast(p=0.5),
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def get_valid_transform():
    return A.Compose([
        ToTensorV2(p=1.0)
    ], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels']))


def collate_fn(batch):
    return tuple(zip(*batch))
