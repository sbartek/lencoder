"""
Transforms categorical column into numbers
"""

from .utils import dfs_column2num_dicts, dict2fun, df_ints2one_hot_encoding

class Encoder:
    
    def __init__(self, colname, dfs):
        self.dfs = dfs
        self.colname = colname

    def create_dicts(self):
        self.item2num, self.num2item = \
          dfs_column2num_dicts(self.colname, *self.dfs)
        return self

    def transform_colummn2int(self, df):
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
