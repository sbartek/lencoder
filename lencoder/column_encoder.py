from .encoder import Encoder

class ColumnEncoder(Encoder):

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

    def encode(self, df):
        df[self.colname] = self.items2nums(df[self.colname])
