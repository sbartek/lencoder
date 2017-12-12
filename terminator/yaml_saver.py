import yaml

class YamlSaver:

    def __init__(self, config=None):
        if config is None:
            self.config = {}
        else:
            self.config = config
        
    def load(self, stream):
        return yaml.load(stream)

    def dump(self, data):
        return yaml.dump(data, default_flow_style=False, encoding=self.encoding)

    @property
    def encoding(self):
        return self.config.get('encoding')

