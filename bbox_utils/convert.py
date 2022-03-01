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
