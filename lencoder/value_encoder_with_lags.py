import pandas as pd
import numpy as np

from .value_encoder import ValueEncoder
from . import time_based_utils

class ValueEncoderWithLags(ValueEncoder):

    def __init__(
            self, df, date_column, group_columns,
            value_columns,
            lags, aggregations, **kwargs):
        self.date_column = date_column
        self.group_columns = group_columns
        self.lags = lags
        groupby_columns = [self.date_column] + self.group_columns
        super()\
            .__init__(df, groupby_columns, value_columns, aggregations, **kwargs)

    def encode(self):
        encoded = super().encode()
        return encoded.add_lags(
            self.date_column, self.group_columns,
            self.lags)

def value_encodigs_with_lags(
        df, date_column, group_columns,
        value_columns, lags, aggregations, add2dt=False, **kwargs):
    venc = ValueEncoderWithLags(df, date_column, group_columns,
        value_columns, lags, aggregations, **kwargs)
    if add2dt:
        return venc.add_encoding()
    return venc.encode()

def add_value_encodigs_with_lags(df, *args, **kwargs):
    return value_encodigs_with_lags(df, *args, add2dt=True, **kwargs)

pd.DataFrame.value_encodigs_with_lags = value_encodigs_with_lags
pd.DataFrame.add_value_encodigs_with_lags = add_value_encodigs_with_lags

