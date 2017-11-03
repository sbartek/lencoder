"""
Transforms categorical column into numbers
"""
import os
import pickle
import numpy as np

from .utils import dict2fun, df_ints2one_hot_encoding

class Encoder:
    """
    Encoder Class: encode and decode a categorical column
    """

    def __init__(self, items=None, colname=None, pickle_fn=None):
        self.items = items
        self.colname = colname
        self.item2num = {}
        self.num2item = {}
        if pickle_fn is None:
            if self.colname is not None:
                pickle_fn = self.colname + ".pickle"
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

    def create_dicts(self):
        """Run it if creating dicts from items"""
        self.item2num, self.num2item = items2num_dicts(self.unique_items)
        return self

    def dump_dicts(self):
        """Save dicts on disk"""
        if not os.path.exists(self.pickle_dir):
            os.makedirs(self.pickle_dir)
        data = {
            'item2num': self.item2num,
            'num2item': self.num2item
        }
        with open(self.pickle_path, 'wb') as file:
            pickle.dump(data, file, pickle.HIGHEST_PROTOCOL)

    def load_dicts(self):
        """Load dicts on disk"""
        with open(self.pickle_path, 'rb') as file:
            data = pickle.load(file)
            self.item2num = data['item2num']
            self.num2item = data['num2item']
        return self
            
    @property
    def unique_items(self):
        """List of unique items"""
        return list(set(self.items))

    def items2nums(self, items):
        item2num_fun = np.vectorize(dict2fun(self.item2num))
        return item2num_fun(items)
    
    def modify_column_2int(self, df):
        item2num_fun = np.vectorize(dict2fun(self.item2num))
        df[self.colname] = item2num_fun(df[self.colname])
        return self

    def one_hot_encoding(self, df):
        max_item_num = max(self.item2num.values())
        return df_ints2one_hot_encoding(df, self.colname, max_item_num)

    def encode(self, df):
        return self.create_dicts()\
          .transform_colummn2int(df)\
          .one_hot_encoding(df)

def items2num_dicts(items):
    """
    input: list of items to encode
    """
    items = list(set(items))
    item2num = {}
    num2item = {}
    for i in range(len(items)):
        item2num[items[i]] = i
        num2item[i] = items[i]
    return item2num, num2item
