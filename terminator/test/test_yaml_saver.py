from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length, has_key

from terminator.yaml_saver import YamlSaver

class TestYamlSaver(TestCase):

    def setUp(self):
        self.encoder_ys = YamlSaver()
        
    def test_load(self):
        stream = """
- Hesperiidae
- Papilionidae
- Apatelodidae
- Epiplemidae
"""
        items = self.encoder_ys.load(stream)
        assert_that(items, has_length(4))

    def test_load_with_eur(self):
        stream = """
1€: 1
ala: 2
"""
        items = self.encoder_ys.load(stream)
        assert_that(items, has_key('1€'))

    def test_dump_load(self):
        data = {'1€': 1, 'ala': 2}
        stream = self.encoder_ys.dump(data)
        items = self.encoder_ys.load(stream)
        assert_that(items, equal_to(data))

    def test_dump_load_with_encoding(self):
        self.encoder_ys.config['encoding'] = 'utf-8'
        data = {'1€': 1, 'ala': 2}
        stream = self.encoder_ys.dump(data)
        items = self.encoder_ys.load(stream)
        assert_that(items, data)

    
