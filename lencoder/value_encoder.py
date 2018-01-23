import pandas as pd
import numpy as np

class ValueEncoder:

    def __init__(self, df, groupby_columns, value_columns, aggregations, **kwargs):

        self.df = df
        self.groupby_columns = groupby_columns
        self.columns = value_columns
        self.aggregations = aggregations
        
        self.config = dict(
            group_value_joiner=':',
            prefix = None
        )
        self.config = dict(self.config, **kwargs)
        
        if self.prefix is None:
            self.prefix = self.generated_prefix

    @property
    def group_value_joiner(self):
        return self.config.get('group_value_joiner')

    @property
    def prefix(self):
        return self.config.get('prefix')

    @prefix.setter
    def prefix(self, new_prefix):
        self.config['prefix'] = new_prefix
        
    @property
    def generated_prefix(self):
        return '_'.join(self.groupby_columns) + self.group_value_joiner
    
    @property
    def renamed_columns(self):
        return { column: self.prefix+column for column in self.columns}

    def encode(self):
        columns = list(self.renamed_columns.values())
        encoded_df = self.df.rename(columns=self.renamed_columns)\
          .groupby(self.groupby_columns, as_index=False)[columns]\
          .agg(self.aggregations).reset_index()    
        level_0 = encoded_df.columns.get_level_values(0)
        level_1 = encoded_df.columns.get_level_values(1)
        encoded_df.columns = encoded_df.columns.droplevel(0)
        for i in range(len(level_0)):
            if  level_1[i] == "":
                encoded_df.columns.values[i] = level_0[i]
            else:
                encoded_df.columns.values[i] = level_0[i]\
                  + self.group_value_joiner + level_1[i]
        return encoded_df

def value_encodigs(df, groupby_columns, columns, aggregations, **kwargs):
    venc = ValueEncoder(df, groupby_columns, columns, aggregations, **kwargs)
    return venc.encode()

pd.DataFrame.value_encodigs = value_encodigs
