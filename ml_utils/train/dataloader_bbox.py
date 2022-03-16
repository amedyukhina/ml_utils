import numpy as np
import pandas as pd
from torch.utils.data import DataLoader

from ..dataset.dataset_object_training import DatasetObjectTraining
from ..transforms.bbox import get_train_transform, get_valid_transform
from ..utils.convert import wh_to_xy
from ..utils.utils import collate_fn


def split_train_val(df, id_name='image_id', val_fraction=0.2, seed=None):
    image_ids = df[id_name].unique()
    nvalid = int(round(val_fraction * len(image_ids)))
    np.random.seed(seed)
    image_ids = np.random.choice(image_ids, len(image_ids), replace=False)

    print(rf"Train images: {len(image_ids) - nvalid}; validation images: {nvalid}")
    valid_df = df[df[id_name].isin(image_ids[-nvalid:])]
    train_df = df[df[id_name].isin(image_ids[:-nvalid])]
    return train_df, valid_df


def get_data_loaders(bbox_fn, input_dir, bbox_fn_val=None, val_fraction=0.2, seed=None,
                     id_name='image_id', shuffle=True, **kwargs):
    dfs = [wh_to_xy(pd.read_csv(fn)) for fn in [bbox_fn, bbox_fn_val] if fn is not None]
    if len(dfs) == 2:
        train_df, valid_df = dfs
    else:
        train_df, valid_df = split_train_val(dfs[0], id_name, val_fraction, seed)

    dls = []
    for cur_df, shf, tr in zip([train_df, valid_df],
                               [shuffle, False],
                               [get_train_transform(), get_valid_transform()]):
        ds = DatasetObjectTraining(cur_df, input_dir, tr)
        dls.append(DataLoader(ds, shuffle=shf, collate_fn=collate_fn, **kwargs))
    return dls
