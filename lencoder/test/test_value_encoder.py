from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, is_

from lencoder.value_encoder import ValueEncoder, value_encodigs

class TestColumnEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'days': sorted(list(range(4)) * 4),
            'group': sorted(['A', 'B'] * 2) * 4,
            'value1': sorted(list(range(8)) * 2),
            'value2': list(range(8)) * 2
        })

    def test_encode_two_columns(self):
        venc = ValueEncoder(
            self.df, ['days'], ['value1', 'value2'],
            aggregations=['mean', 'sum'])
        encoded_df = venc.encode()
        target_df = pd.DataFrame(
                   {'days:value1:sum': [2, 10, 18, 26]})
        assert_that(encoded_df[['days:value1:sum']]\
                    .equals(target_df), is_(True))

    def test_encode_two_groups(self):
        venc = ValueEncoder(
            self.df, ['days', 'group'], ['value1', 'value2'],
            aggregations=['mean', 'sum'])
        encoded_df = venc.encode()
        target_df = pd.DataFrame(
            {'days_group:value1:sum': list(range(0, 16, 2))})
        assert_that(encoded_df[['days_group:value1:sum']]\
                    .equals(target_df), is_(True))

    def test_value_encodigs(self):
        encoded_df = self.df.value_encodigs(
            ['days'], ['value1', 'value2'],
            aggregations=['mean', 'sum'])
        target_df = pd.DataFrame(
                   {'days:value1:sum': [2, 10, 18, 26]})
        assert_that(encoded_df[['days:value1:sum']]\
                    .equals(target_df), is_(True))
