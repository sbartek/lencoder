from unittest import TestCase
import numpy as np
import pandas as pd

from hamcrest import assert_that, equal_to, is_

from lencoder.time_based_utils import add_lags

class TestAddLags(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({
            'days': sorted(list(range(5)) * 2),
            'group': ['A', 'B'] * 5,
            'value1': sorted(list(range(5)) * 2),
            'value2': list(range(10))
        })

    def test_add_lags(self):
        lagged_df = self.df.add_lags(
            'days', ['group'], lags=[1, 2]
        )
        target_df = pd.DataFrame({
            'value1_1': [float(i) for i in ([0, 0] + sorted(list(range(4)) * 2))]
        })
        assert_that(lagged_df[['value1_1']].equals(target_df), is_(True))

    def test_add_lags_one_column(self):
        lagged_df = self.df.add_lags(
            'days', ['group'], lags=[1, 2], columns2lag=['value1']
        )
        target_df = pd.DataFrame({
            'value1_1': [float(i) for i in ([0, 0] + sorted(list(range(4)) * 2))]
        })
        assert_that(lagged_df[['value1_1']].equals(target_df), is_(True))

    
