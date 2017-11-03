import pandas as pd

from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length

from terminator import Encoder

class TestEncoder(TestCase):

    def setUp(self):
        self.df = pd.DataFrame({'a': range(10), 'b': list(range(2))*5})
        
    def test_encoder(self):
        encoder = Encoder('b', [self.df])
        assert_that(True, equal_to(False))
