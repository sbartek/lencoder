import os
import shutil
from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, has_length, has_key, has_value,\
    has_item, not_none, none, is_not

from terminator.one_hot_encoder import OneHotEncoder

class TestOneHotEncoder(TestCase):

    def test_init(self):
        items = ["ma", "€"]
        ohenc = OneHotEncoder(items).create_dicts()
        assert_that(list(ohenc.items),
                    equal_to(['<NAN>', 'ma', '€']))
        ohe = ohenc.encode(np.array(["ma", "€", "1"]))
        np.testing.assert_array_equal(
            ohe,
            np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]])
        )
