import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from ..utils.utils import load_image


class DatasetObjectInference(Dataset):
    """
    Dataset class for prediction of bounding boxes.
    """

    def __init__(self, dataframe, image_dir, transforms=None, maxsize=None):
        """
        Initialize the Dataset instance.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe with a list of image names
        image_dir : str
            Input directory with the images.
        transforms : transform
            Transform to apply before predictions.
            Default is None
        """
        super().__init__()

        self.image_ids = dataframe['image_id'].unique()
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.maxsize = maxsize

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image = load_image(f'{self.image_dir}/{image_id}', self.maxsize)

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id

    def __len__(self) -> int:
        return self.image_ids.shape[0]


class DatasetObjectInferenceMosaic(DatasetObjectInference):
    """
    Dataset class for prediction of bounding boxes in mosaic images.
    """

    def __init__(self, dataframe, image_dir, transforms=None, maxsize=None):
        """
        Initialize the Dataset instance.

        Parameters
        ----------
        dataframe : pd.DataFrame
            Dataframe with a list of image names
        image_dir : str
            Input directory with the images.
        transforms : transform
            Transform to apply before predictions.
            Default is None
        """
        super().__init__(dataframe, image_dir, transforms, maxsize)

    def __getitem__(self, index: int):
        image_id = self.image_ids[index]

        image = load_image(f'{self.image_dir}/{image_id}', maxsize=None)
        x1, x2, y1, y2 = self.df.iloc[index][['x1', 'x2', 'y1', 'y2']].values
        image = image[y1:y2, x1:x2]
        image = np.pad(image, [(0, self.maxsize - image.shape[0]),
                               (0, self.maxsize - image.shape[1]), (0, 0)])

        if self.transforms:
            sample = {
                'image': image,
            }
            sample = self.transforms(**sample)
            image = sample['image']

        return image, image_id, (x1, y1)
