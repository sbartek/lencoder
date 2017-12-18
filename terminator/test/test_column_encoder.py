import os
import shutil
from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, has_length, has_key, has_value, has_item

from terminator.column_encoder import ColumnEncoder
from terminator.encoder import has_nans

class TestColumnEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a': range(10), 'b': list(range(1, 3))*5})
        self.b_encoder = ColumnEncoder(items=self.df['b'], colname='b').create_dicts()

    def test_create_dicts(self):
        self.b_encoder.create_dicts()
        assert_that(self.b_encoder.item2num, has_key('2'))
        assert_that(self.b_encoder.item2num, has_value(0))
        assert_that(self.b_encoder.num2item, has_key(0))
        assert_that(self.b_encoder.num2item, has_value('2'))

    def test_items2num(self):
        self.b_encoder.create_dicts()
        assert_that(
            self.b_encoder.items2nums(self.df['b']), has_length(10))

    def test_encode(self):
        print(self.df['b'])
        self.b_encoder.create_dicts()
        self.b_encoder.encode(self.df)
        print(self.df['b'])
        assert_that(self.df['b'], has_item(0))

    def test_dump_and_load_dicts(self):
        self.b_encoder.create_dicts()
        self.b_encoder.dump_dicts("b_df_")
        new_encoder = ColumnEncoder.create_from_saved_dicts("b_df_")
        print(new_encoder.item2num)
        assert_that(new_encoder.item2num, has_key('2'))
        assert_that(new_encoder.item2num, has_value(0))
        assert_that(new_encoder.num2item, has_key(0))
        assert_that(new_encoder.num2item, has_value('2'))
        new_encoder.encode(self.df)
        assert_that(self.df['b'], has_item(0))

