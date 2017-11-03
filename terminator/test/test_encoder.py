import os
import shutil
import pandas as pd

from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length, has_key, has_value

from terminator import Encoder

class TestEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a': range(10), 'b': list(range(1, 3))*5})
        self.b_encoder = Encoder(self.df['b'], 'b')
        self.b_encoder.create_dicts()

    def test_create_dicts(self):
        assert_that(self.b_encoder.item2num, has_key(2))
        assert_that(self.b_encoder.item2num, has_value(0))
        assert_that(self.b_encoder.num2item, has_key(0))
        assert_that(self.b_encoder.num2item, has_value(2))

    def test_items2num(self):
        assert_that(
            self.b_encoder.items2nums(self.df['b']), has_length(10))

    def test_default_pickle_path(self):
        assert_that(self.b_encoder.pickle_path, equal_to("pickles/b.pickle"))

    def test_dump_and_load_dicts(self):
        if os.path.exists(self.b_encoder.pickle_dir):
            shutil.rmtree(self.b_encoder.pickle_dir)
        assert_that(os.path.exists(self.b_encoder.pickle_dir), equal_to(False))
        self.b_encoder.dump_dicts()
        assert_that(os.path.exists(self.b_encoder.pickle_path), equal_to(True))
        new_encoder = Encoder(colname='b').load_dicts()
        assert_that(new_encoder.item2num, has_key(2))
        assert_that(new_encoder.item2num, has_value(0))
        assert_that(new_encoder.num2item, has_key(0))
        assert_that(new_encoder.num2item, has_value(2))
