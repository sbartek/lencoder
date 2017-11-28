import os
import shutil
from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, has_length, has_key, has_value, has_item

from terminator import ColumnEncoder
from terminator.encoder import has_nans

class TestColumnEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a': range(10), 'b': list(range(1, 3))*5})
        self.b_encoder = ColumnEncoder(self.df['b'], 'b')

    def test_create_dicts(self):
        self.b_encoder.create_dicts()
        assert_that(self.b_encoder.item2num, has_key(2))
        assert_that(self.b_encoder.item2num, has_value(0))
        assert_that(self.b_encoder.num2item, has_key(0))
        assert_that(self.b_encoder.num2item, has_value(2))

    def test_items2num(self):
        self.b_encoder.create_dicts()
        assert_that(
            self.b_encoder.items2nums(self.df['b']), has_length(10))

    def test_modify_column_item2num(self):
        self.b_encoder.create_dicts()
        self.b_encoder.modify_column_item2num(self.df)
        assert_that(
            self.df['b'], has_item(0))

    def test_encode(self):
        self.b_encoder.encode(self.df)
        assert_that(self.df['b'], has_item(0))

    def test_default_pickle_path(self):
        assert_that(self.b_encoder.pickle_path, equal_to("pickles/b.pickle"))

    def test_dump_and_load_dicts(self):
        self.b_encoder.create_dicts()
        if os.path.exists(self.b_encoder.dicts_saver.pickle_dir):
            shutil.rmtree(self.b_encoder.dicts_saver.pickle_dir)
        assert_that(
            os.path.exists(self.b_encoder.dicts_saver.pickle_dir),
            equal_to(False))
        self.b_encoder.dump_dicts()
        assert_that(os.path.exists(self.b_encoder.pickle_path), equal_to(True))
        new_encoder = ColumnEncoder(colname='b').load_dicts()
        assert_that(new_encoder.item2num, has_key(2))
        assert_that(new_encoder.item2num, has_value(0))
        assert_that(new_encoder.num2item, has_key(0))
        assert_that(new_encoder.num2item, has_value(2))
        new_encoder.encode(self.df)
        assert_that(self.df['b'], has_item(0))

class TestHasNans(TestCase):

    def test_list(self):
        assert_that(has_nans([0, 3, np.nan, 3]), equal_to(True))
        assert_that(has_nans([0, 3, 3]), equal_to(False))

    def test_dataframe_column(self):
        assert_that(
            has_nans(pd.DataFrame({'a': [0, 3, np.nan, 3]})['a']),
            equal_to(True))
        assert_that(
            has_nans(pd.DataFrame({'a': [0, 3, 3]})['a']),
            equal_to(False))
        
class TestColumnEncoderWithNans(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'b': [0, 3, np.nan, 3]})

    def test_rise_value_error(self):
        items = self.df['b']
        with self.assertRaises(ValueError):
            ColumnEncoder(items, 'b')
