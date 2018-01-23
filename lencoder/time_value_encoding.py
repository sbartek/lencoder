import pandas as pd
import numpy as np

def value_encodigs_with_lags(
        df, date_column, group_columns, value_columns, lags, aggregations=None):
    prefix = "_".join([date_column] + group_columns) + "_"
    return df.add_value_encodigs(
        [date_column] + group_columns, value_columns, aggregations=aggregations, prefix=prefix
    ).add_lags(date_column, group_columns, lags=lags)

pd.DataFrame.value_encodigs_with_lags = value_encodigs_with_lags

def add_value_encodigs_with_lags(
        df, date_column, group_columns, value_columns, lags, aggregations=None):
    return df.merge(
        df.value_encodigs_with_lags(
            date_column, group_columns, value_columns, lags, aggregations),
        on=[date_column] + group_columns, how='left')

pd.DataFrame.add_value_encodigs_with_lags = add_value_encodigs_with_lags
