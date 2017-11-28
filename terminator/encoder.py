"""
Transforms categorical column into numbers
"""
import os
import pickle
import numpy as np
import pandas as pd

from .utils import dict2fun

class EncoderDictsSaver:
    """Class for pickling encoders"""
    def __init__(self, encoder, pickle_fn):
        self.encoder = encoder
        if pickle_fn is None:
            if self.encoder.colname is not None:
                pickle_fn = self.encoder.colname + ".pickle"
            else:
                pickle_fn = "encoder_dicts.pickle"
            self.pickle_dir = "pickles"
        else:
            self.pickle_dir = "."
        self.pickle_fn = pickle_fn

    @property
    def pickle_path(self):
        """Generate full path to pickle"""
        return os.path.join(self.pickle_dir, self.pickle_fn)

    def dump_dicts(self):
        """Save dicts on disk"""
        if not os.path.exists(self.pickle_dir):
            os.makedirs(self.pickle_dir)
        data = {
            'item2num': self.encoder.item2num,
            'num2item': self.encoder.num2item
        }
        with open(self.pickle_path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def load_dicts(self):
        """Load dicts on disk"""
        with open(self.pickle_path, 'rb') as file:
            data = pickle.load(file)
            self.encoder.item2num = data['item2num']
            self.encoder.num2item = data['num2item']
        return self

class ColumnEncoder:
    """
    Encode and decode a categorical column
    """

    def __init__(self, items=None, colname=None, pickle_fn=None, replace_nans_with=np.nan):
        if items is not None:
            if np.isnan(replace_nans_with):
                try:
                    assert not has_nans(items)
                except AssertionError:
                    raise ValueError(
                        "items contains nan, add parameter replace_nans_with to make it work")

            self.items = items[:]
            if not np.isnan(replace_nans_with):
                self.items = replace_nans(items, replace_nans_with)
        else:
            self.items = None
        self.colname = colname
        self.item2num = None
        self.num2item = None
        self.dicts_saver = EncoderDictsSaver(self, pickle_fn)

    def create_dicts(self):
        """Run it if creating dicts from items"""
        self.item2num, self.num2item = items2num_dicts(self.unique_items)
        return self

    def dump_dicts(self):
        """Save dicts on disk"""
        self.dicts_saver.dump_dicts()
        return self

    def load_dicts(self):
        """Load dicts on disk"""
        self.dicts_saver.load_dicts()
        return self
            
    @property
    def unique_items(self):
        """List of unique items"""
        return list(set(self.items))

    def items2nums(self, items):
        item2num_fun = np.vectorize(dict2fun(self.item2num))
        return item2num_fun(items)
    
    def modify_column_item2num(self, df):
        df.loc[:, self.colname] = self.items2nums(df[self.colname])
        return df

    def encode(self, df):
        if self.item2num is None:
            self.create_dicts()
        return self.modify_column_item2num(df)

    @property
    def pickle_path(self):
        """Get full path to pickle"""
        return self.dicts_saver.pickle_path


class ColumnOneHotEncoder(ColumnEncoder):

    def one_hot_encoding(self, df):
        max_num = max(self.item2num.values())
        return one_hot_encoding_eye(
            df[self.colname], max_num, colname=self.colname + "_")

    def add_one_hot_encoding_columns(self, df):
        new_df = self.one_hot_encoding(df)
        return concatenate_dfs_on_pseudo_index(df, new_df)
    
    def encode(self, df):
        df = super().encode(df)
        df = self.add_one_hot_encoding_columns(df)
        df = df.drop(self.colname, axis=1)
        return df
          
def items2num_dicts(items):
    """
    input: list of items to encode
    """
    items = list(set(items))
    item2num = {}
    num2item = {}
    for i, _ in enumerate(items):
        item2num[items[i]] = i
        num2item[i] = items[i]
    return item2num, num2item

def one_hot_encoding_eye(nums, max_num=None, colname="num_", dtype=np.bool):
    if max_num is None:
        max_num = max(nums)
    return pd.DataFrame(np.eye(max_num + 1, dtype=dtype)[nums])\
      .rename(columns=lambda x: colname+str(x))

def has_nans(items):
    for item in items:
        if np.isnan(item):
            return True
    return False

def replace_nans(items, replace_nans_with):
    for i in range(len(items)):
        if np.isnan(item[i]):
            item[i] = replace_nans_with
    return items

def add_pseudoindex(df, pseudoindex):
    df.loc[:, pseudoindex] = range(df.shape[0])
    
def drop_pseudoindex(df, pseudoindex):
    df.drop(pseudoindex, axis=1, inplace=True)
    
def merge_on_pseudoindex(df1, df2, pseudoindex):
    return df1.merge(df2, on=pseudoindex)

def concatenate_dfs_on_pseudo_index(df1, df2, pseudoindex="pseudoindex___"):
    add_pseudoindex(df1, pseudoindex)
    add_pseudoindex(df2, pseudoindex)
    final_df = merge_on_pseudoindex(df1, df2, pseudoindex).copy()
    drop_pseudoindex(final_df, pseudoindex)
    return final_df

