import pandas as pd
import numpy as np


def merge_with_lag(df1, df2, lag, date_column, on_columns=None):
    """
    df1: data frame with which we left join results
    df2: data frame where we will apply lag
    date_column: column that stores 'dates' where we apply lag, 
                 should support operation 
       df2[date_column] + lag
    on_columns: columns that on which we apply join
    """
    if on_columns is None:
        on_columns = [] 
    left_on = (df1[date_column], )
    right_on = (df2[date_column]+lag, )
    to_drop = ['{}_{}'.format(date_column, lag)]
    for column in on_columns:
        left_on += (df1[column], )
        right_on += (df2[column], )
        to_drop += ['{}_{}'.format(column, lag)] 
    return df1.merge(df2,
                     left_on=left_on, right_on=right_on,
                     how='left', suffixes=["", "_{}".format(lag)]
                    ).fillna(0).drop(to_drop, axis=1)

pd.DataFrame.merge_with_lag = merge_with_lag

def add_lags(df, date_column, group_columns, lags, columns2lag=None):
    if columns2lag is None:
        laged_df = df
    else:
        laged_df = df[[date_column] + group_columns + columns2lag]
    for lag in lags:
        laged_df = laged_df.merge_with_lag(
            df, lag, date_column=date_column, on_columns=group_columns)
    return laged_df

pd.DataFrame.add_lags = add_lags
