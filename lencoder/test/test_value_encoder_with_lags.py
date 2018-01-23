from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, is_

from lencoder.value_encoder_with_lags import ValueEncoderWithLags, value_encodigs_with_lags

class TestColumnEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'days': sorted(list(range(4)) * 4),
            'group': sorted(['A', 'B'] * 2) * 4,
            'value1': sorted(list(range(8)) * 2),
            'value2': list(range(8)) * 2
        })

    def test_encode_column(self):
        venc = ValueEncoderWithLags(
            self.df, 'days', ['group'], ['value1'],
            lags = [1, 2], aggregations=['sum'])
        encoded_df = venc.encode()
        target_df = pd.DataFrame(
                   {'days_group:value1:sum_1':
                    [0., 0., 0.0, 2., 4., 6., 8., 10.]})
        assert_that(encoded_df[['days_group:value1:sum_1']]\
                    .equals(target_df), equal_to(True))

    def test_value_encodigs_with_lags(self):
        encoded_df = self.df.value_encodigs_with_lags(
            'days', ['group'], ['value1'],
            lags = [1, 2], aggregations=['sum']
        )
        target_df = pd.DataFrame(
                   {'days_group:value1:sum_1':
                    [0., 0., 0.0, 2., 4., 6., 8., 10.]})
        assert_that(encoded_df[['days_group:value1:sum_1']]\
                    .equals(target_df), equal_to(True))

