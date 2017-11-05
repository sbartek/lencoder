import os
import shutil
from unittest import TestCase
import pandas as pd

from hamcrest import assert_that, equal_to, has_length, has_key, has_value,\
    has_item, not_none, none, is_not

from terminator import ColumnEncoder, ColumnOneHotEncoder
from terminator.encoder import one_hot_encoding_eye

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

class TestOneHotEncodingFunction(TestCase):

    def test_one_hot_encoding_eye(self):
        df = one_hot_encoding_eye([0, 1, 0, 2], max_num=2, colname="a")
        assert_that(df["a0"][0], equal_to(True))
        assert_that(df["a0"][1], equal_to(False))
        assert_that(df["a2"][3], equal_to(True))
        
class TestColumnOneHotEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a': range(10), 'b': list(range(1, 3))*5})
        self.b_ohencoder = ColumnOneHotEncoder(self.df['b'], 'b')

    def test_dicts_are_none(self):
        assert_that(self.b_ohencoder.item2num, none())

    def test_dicts_in_not_none(self):
        self.b_ohencoder.create_dicts()
        assert_that(self.b_ohencoder.item2num, not_none())

    def test_dicts_in_not_none2(self):
        self.b_ohencoder.create_dicts().modify_column_item2num(self.df)
        assert_that(self.b_ohencoder.item2num, not_none())

    def test_one_hot_encoding(self):
        self.b_ohencoder.create_dicts().modify_column_item2num(self.df)
        ohe = self.b_ohencoder.one_hot_encoding(self.df)
        assert_that(ohe['b_0'], has_item(True))

    def test_add_one_hot_encoding_columns(self):
        self.b_ohencoder.create_dicts().modify_column_item2num(self.df)
        df = self.b_ohencoder.add_one_hot_encoding_columns(self.df)
        assert_that(self.df.columns.values, has_item('b_0'))

    def test_encode(self):
        self.b_ohencoder.encode(self.df)
        assert_that(self.df.columns.values, has_item('b_0'))
        assert_that(self.df.columns.values, is_not(has_item('b')))
