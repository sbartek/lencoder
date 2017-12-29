from unittest import TestCase
from hamcrest import assert_that, equal_to, has_length, has_key

from lencoder.yaml_saver import YamlSaver

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

    def test_dump2file_and_load_from_file(self):
        data = {'1€': 1, 'ala': 2}
        file_name = "lencoder/test/yamls/ala_ma_kota.yaml"
        self.encoder_ys.dump2file(data, file_name)
        new_data = self.encoder_ys.load_from_file(file_name)
        assert_that(data, equal_to(new_data))
