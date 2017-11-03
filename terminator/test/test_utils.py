from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length

import pandas as pd

class TestStringMethods(TestCase):

    def setUp(self):
        self.df0 = pd.DataFrame({'a': range(10)})
        self.df1 = pd.DataFrame({'a': range(4, 13)})
        self.df2 = pd.DataFrame({'a': range(15, 17)})

    def test_dfs_column2list_uniques(self):
        #a_list = dfs_column2list_uniques('a', self.df0, self.df1, self.df2)
        #assert_that(a_list, has_length(15))
        pass
