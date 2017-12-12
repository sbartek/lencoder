import os
import yaml

class YamlSaver:

    def __init__(self, config=None):
        if config is None:
            self.config = {}
        else:
            self.config = config
        
    def load(self, stream):
        return yaml.load(stream)

    def dump(self, data, stream=None):
        return yaml.dump(
            data,
            stream=stream,
            default_flow_style=False,
            encoding=self.encoding)

    @property
    def encoding(self):
        return self.config.get('encoding')

    @property
    def dir_name(self):
        return self.config.get('dir_name', ".")

    def full_path(self, file_name):
        """Generate full path to yaml file"""
        return os.path.join(self.dir_name, file_name)
    
    def dump2file(self, data, file_name):
        try:
            with open(self.full_path(file_name), 'w') as yaml_file:
                self.dump(data, yaml_file)
        except FileNotFoundError as exception:
            if not os.path.exists(self.dir_name):
                os.makedirs(self.dir_name)
                self.dump2file(data, file_name)
            else:
                raise exception

    def load_from_file(self, file_name):
        with open(self.full_path(file_name), 'r') as yaml_file:
            data = self.load(yaml_file)
        return data
