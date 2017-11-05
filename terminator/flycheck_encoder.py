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
    Encoder Class: encode and decode a categorical column
    """

    def __init__(self, items=None, colname=None, pickle_fn=None):
        self.items = items
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
        df[self.colname] = self.items2nums(df[self.colname])
        return self

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
        for column in new_df.columns.values:
            df[column] = new_df[column]
        return df
    
    def encode(self, df):
        super().encode(df).add_one_hot_encoding_columns(df)
        df.drop(self.colname, axis=1, inplace=True)
        return self
          
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
