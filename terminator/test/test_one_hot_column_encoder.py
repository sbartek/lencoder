import os
import shutil
from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, has_length, has_key, has_value, has_item

from terminator.one_hot_column_encoder import ColumnOneHotEncoder
from terminator.encoder import has_nans

class TestOneHotColumnEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a': range(10), 'b': list(range(1, 3))*5})
        self.b_encoder = ColumnOneHotEncoder(items=self.df['b'], colname='b')\
          .create_dicts()

    def test_items2nums(self):
        self.b_encoder.create_dicts()
        assert_that(
            self.b_encoder.items2nums(self.df['b']).shape, equal_to((10, 3)))

    def test_encode(self):
        self.b_encoder.create_dicts()
        self.b_encoder.encode(self.df)
        print(self.b_encoder.encode(self.df))
        assert_that(1, equal_to(2))
