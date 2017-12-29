import pandas as pd

from .one_hot_encoder import OneHotEncoder

class ColumnOneHotEncoder(OneHotEncoder):

    def __init__(self, items=None, colname=None, config=None, **kwargs):
        super().__init__(items=items, config=config)
        if colname is not None:
            self.colname = colname

    @property
    def colname(self):
        return self.config.get('colname')

    @colname.setter
    def colname(self, colname):
        self.config['colname'] = colname

    def items2nums(self, items2encode):
        return super().encode(items2encode)

    def encode(self, df, drop_column=True):
        new_df =  pd.DataFrame(super().encode(df[self.colname]))\
            .rename(columns=lambda x: self.colname+"_"+str(x))
        if drop_column:
            df = df.drop(self.colname, axis=1)
        return concatenate_dfs_on_pseudo_index(df, new_df)

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
