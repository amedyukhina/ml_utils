import os
import unittest
from ddt import ddt
import numpy as np

import pandas as pd
from ..convert import wh_to_xy, xy_to_wh

INPUT_DIR = os.path.dirname(os.path.abspath(__file__)) + '/../../example_data'


@ddt
class TestConvert(unittest.TestCase):

    def test_convert(self):
        df = pd.read_csv(os.path.join(INPUT_DIR, 'bboxes.csv'))
        df2 = wh_to_xy(df)
        df2 = xy_to_wh(df2)
        for c in df2.columns:
            self.assertSequenceEqual(list(np.array(df[c])), list(np.array(df2[c])))


if __name__ == '__main__':
    unittest.main()
