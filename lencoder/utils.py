#import sys
#print(sys.modules.keys())

import pandas as pd
import numpy as np

def dict2fun(dictionary):
    def fun(key):
        return dictionary[key]
    return fun

def dict2fun_with_nan_replacement(dictionary, nan_replacement):
    def fun(key):
        return dictionary.get(key, dictionary[nan_replacement])
    return fun

def add_value_encodigs(df, groupby_columns, columns, aggregations=None, prefix=None):

    if aggregations is None:
        ## aggregations = ['mean', 'max', 'min', 'std', 'median'] 
        aggregations = ['mean', 'std', 'median'] 
    if prefix is not None:
        rename_columns = { column: prefix+column for column in columns}
        columns = list(rename_columns.values())
    else:
        rename_columns = {}
        
    encoded_df = df.rename(columns=rename_columns)\
        .groupby(groupby_columns, as_index=False)[columns]\
        .agg(aggregations).reset_index()    
    level_0 = encoded_df.columns.get_level_values(0)
    level_1 = encoded_df.columns.get_level_values(1)
    encoded_df.columns = encoded_df.columns.droplevel(0)
    for i in range(len(level_0)):
        if  level_1[i] == "":
            encoded_df.columns.values[i] = level_0[i]
        else:
            encoded_df.columns.values[i] = level_0[i] + "_" + level_1[i]
    return encoded_df

pd.DataFrame.add_value_encodigs = add_value_encodigs



def merge_with_lag(df1, df2, lag, date_column, on_columns=None):
    """
    df1: data frame with which we left join results
    df2: data frame where we will apply lag
    date_column: column that stores 'dates' where we apply lag, should support operation df2[date_column] + lag
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
    return df1.merge(
            df2, left_on=left_on, right_on=right_on, how='left', suffixes=["", "_{}".format(lag)]
        ).fillna(0)\
        .drop(to_drop, axis=1)
    
pd.DataFrame.merge_with_lag = merge_with_lag

def add_lags(df, date_column, on_columns, lags):
    laged_df = df
    for lag in lags:
        laged_df = laged_df.merge_with_lag(df, lag, date_column=date_column, on_columns=on_columns)
    return laged_df

pd.DataFrame.add_lags = add_lags
