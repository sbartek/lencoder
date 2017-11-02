import pandas as pd
import numpy as np

def items2num_dicts(items):
    items = list(set(items))
    item2num = {}
    num2item = {}
    for i in range(len(items)):
        item2num[items[i]] = i
        num2item[i] = items[i]
    return item2num, num2item

def one_hot_encoding(nums, max_num=None, colname="num_", dtype=np.bool):
    if max_num is None:
        max_num = max(nums)
    return pd.DataFrame(np.eye(max_num + 1, dtype=dtype)[nums])\
      .rename(columns=lambda x: colname+str(x))

def dict2fun(dictionary):
    def fun(key):
        return dictionary[key]
    return fun

def df_ints2one_hot_encoding(df, colname, max_num):
    new_df = one_hot_encoding(df[colname], max_num=max_num, colname=colname+"_")
    df = pd.concat([df, new_df], axis=1)
    df.drop(colname, axis=1, inplace=True)
    return df


class DataTransformer:
    
    def __init__(self):
        pass
    
class Encoder:
    
    def __init__(self, df, colname):
        self.df = df
        self.colname = colname
