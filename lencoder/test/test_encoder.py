import os
import shutil
from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, has_length, has_key, has_value,\
    has_item, not_none, none, is_not

from lencoder import ColumnEncoder, ColumnOneHotEncoder
from lencoder.encoder import Encoder

# class TestOneHotEncodingFunction(TestCase):

#     def test_one_hot_encoding_eye(self):
#         df = one_hot_encoding_eye([0, 1, 0, 2], max_num=2, colname="a")
#         assert_that(df["a0"][0], equal_to(True))
#         assert_that(df["a0"][1], equal_to(False))
#         assert_that(df["a2"][3], equal_to(True))

class TestEncoder(TestCase):

    def test_init(self):
        items = ["ala", "ma", "kota", "ala", "€"]
        enc = Encoder(items)
        assert_that(list(enc.items), equal_to(["<NAN>", "ala", "kota", "ma", "€"]))

    def test_init_with_nan(self):
        items = ["ala", "ma", "kota", "ala", None, None, np.nan]
        enc = Encoder(items, nan_replacement = "<NAN>")
        assert_that(list(enc.items), equal_to(["<NAN>", "ala", "kota", "ma"]))

    def test_encode(self):
        items = ["ala", "ma", "kota", "ala", None, None, np.nan]
        enc = Encoder(items).create_dicts()
        assert_that(list(enc.encode(np.array(["ala", "ma"]))), equal_to([1, 3]))

    def test_encode_with_non_existent(self):
        items = ["ala", "ma", "kota", "ala", None, None, np.nan]
        enc = Encoder(items).create_dicts()
        assert_that(list(enc.encode(np.array(["ala", "ma", 'psa']))), equal_to([1, 3, 0]))

    def test_dcode_with_non_existent(self):
        items = ["ala", "ma", "kota", "ala", None, None, np.nan]
        enc = Encoder(items).create_dicts()
        assert_that(list(enc.decode(np.array([1, 3, 0]))), equal_to(["ala", "ma", '<NAN>']))

    def test_encode_with_euro(self):
        items = ["ala", "ma", "1€", "ala", None, None, np.nan]
        enc = Encoder(items).create_dicts()
        assert_that(list(enc.encode(np.array(["ala", "ma", 'psa', '1€']))), equal_to([2, 3, 1, 0]))

    def test_dcode_with_eur(self):
        items = ["ala", "ma", "1€", "ala", None, None, np.nan]
        enc = Encoder(items).create_dicts()
        assert_that(list(enc.decode(np.array([1, 3, 0]))), equal_to(['<NAN>', "ma", '1€']))

    def test_dump_load(self):
        items = ["ala", "ma", "1€", "ala", None, None, np.nan]
        enc1 = Encoder.create_from_items_list(items)
        enc1.dump_dicts(prefix="abc_")
        enc2 = Encoder.create_from_saved_dicts(prefix="abc_")
        assert_that(enc2.item2num, equal_to(enc1.item2num))
        assert_that(enc2.num2item, equal_to(enc1.num2item))
        items2 = np.array(list(enc2.item2num.keys()))
        np.testing.assert_array_equal(enc2.encode(items2), enc1.encode(items2))
        nums2 = np.array(list(enc2.item2num.values()))
        np.testing.assert_array_equal(enc2.decode(nums2), enc1.decode(nums2))

class TestCreateEncoderFromItemsList(TestCase):

    def test_encode_with_non_existent(self):
        items = ["ala", "ma", "kota", "ala", None, None, np.nan]
        enc = Encoder.create_from_items_list(items)
        assert_that(list(enc.encode(np.array(["ala", "ma", 'psa']))), equal_to([1, 3, 0]))


