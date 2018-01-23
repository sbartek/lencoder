import pandas as pd
import numpy as np

from .value_encoder import ValueEncoder

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
        print(self.df)
        return encoded.add_lags(
            self.date_column, self.group_columns,
            self.lags)

def value_encodigs_with_lags(
        df, date_column, group_columns,
        value_columns, lags, aggregations, **kwargs):
    venc = ValueEncoderWithLags(df, date_column, group_columns,
        value_columns, lags, aggregations, **kwargs)
    return venc.encode()

pd.DataFrame.value_encodigs_with_lags = value_encodigs_with_lags
