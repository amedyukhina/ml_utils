import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..dataset.dataset_object_training import DatasetObjectTraining
from ..transforms.bbox import get_train_transform, get_valid_transform
from ..utils.convert import wh_to_xy
from ..utils.utils import collate_fn


def get_data_loaders(bbox_fn, input_dir, val_fraction=0.2, seed=None,
                     id_name='image_id', shuffle=True, **kwargs):
    df = pd.read_csv(bbox_fn)
    df = wh_to_xy(df)
    image_ids = df[id_name].unique()
    nvalid = int(round(val_fraction * len(image_ids)))
    np.random.seed(seed)
    image_ids = np.random.choice(image_ids, len(image_ids), replace=False)

    print(rf"Train images: {len(image_ids) - nvalid}; validation images: {nvalid}")
    valid_ids = image_ids[-nvalid:]
    train_ids = image_ids[:-nvalid]

    dls = []
    for ids, shf, tr in zip([train_ids, valid_ids],
                            [shuffle, False],
                            [get_train_transform(), get_valid_transform()]):
        cur_df = df[df[id_name].isin(ids)]
        ds = DatasetObjectTraining(cur_df, input_dir, tr)
        dls.append(DataLoader(ds, shuffle=shf, collate_fn=collate_fn, **kwargs))
    return dls