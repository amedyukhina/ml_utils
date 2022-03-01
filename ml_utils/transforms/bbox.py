import albumentations as A
import torch
from albumentations.pytorch.transforms import ToTensorV2


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
