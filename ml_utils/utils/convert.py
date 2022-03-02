import numpy as np


def wh_to_xy(df):
    if len({'x', 'y', 'width', 'height'}.difference(set(df.columns))) == 0:
        df = df.rename(columns={'x': 'x1', 'y': 'y1', 'width': 'x2', 'height': 'y2'})
        df['x2'] = df['x1'] + df['x2']
        df['y2'] = df['y1'] + df['y2']
    elif len({'x1', 'y1', 'x2', 'y2'}.difference(set(df.columns))) == 0:
        pass
    else:
        raise ValueError("Input dataframe must contain either ['x', 'y', 'width', 'height'] "
                         "columns or [x1', 'y1', 'x2', 'y2] columns")
    return df


def xy_to_wh(df):
    if len({'x', 'y', 'width', 'height'}.difference(set(df.columns))) == 0:
        pass
    elif len({'x1', 'y1', 'x2', 'y2'}.difference(set(df.columns))) == 0:
        df = df.rename(columns={'x1': 'x', 'y1': 'y', 'x2': 'width', 'y2': 'height'})
        df['width'] = df['width'] - df['x']
        df['height'] = df['height'] - df['y']

    else:
        raise ValueError("Input dataframe must contain either ['x', 'y', 'width', 'height'] "
                         "columns or [x1', 'y1', 'x2', 'y2] columns")

    return df


def convert_to_fixed_box_size(data, box_size):
    """
    Convert a list of bounding boxes to bounding boxes of fixed size.

    Parameters
    ----------
    data : pd.DataFrame
        Dataframe with bounding box coordinates (in pixels).
    box_size : scalar
        Desired bounding box size (width=height) in pixels.

    Returns
    -------
    pd.DataFrame
        Dataframe with converted bounding box sizes.
    """
    data['width'] = data['x2'] - data['x1']
    data['height'] = data['y2'] - data['y1']
    data['x1'] = np.int_(data['x1'] + data['width'] / 2 - box_size / 2)
    data['y1'] = np.int_(data['y1'] + data['height'] / 2 - box_size / 2)
    data['x2'] = np.int_(data['x2'] - data['width'] / 2 + box_size / 2)
    data['y2'] = np.int_(data['y2'] - data['height'] / 2 + box_size / 2)
    return data
