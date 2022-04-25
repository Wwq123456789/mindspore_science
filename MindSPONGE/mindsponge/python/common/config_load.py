import yaml
from pprint import pprint, pformat
class Config:
    """
    Configuration namespace. Convert dictionary to members.
    """
    def __init__(self, cfg_dict):
        for k, v in cfg_dict.items():
            if isinstance(v, (list, tuple)):
                setattr(self, k, [Config(x) if isinstance(x, dict) else x for x in v])
            else:
                setattr(self, k, Config(v) if isinstance(v, dict) else v)

    def __str__(self):
        return pformat(self.__dict__)

    def __repr__(self):
        return self.__str__()

def load_config(path):
    """
    Convert yaml file to Obj.
    """
    f = open(path,'r')
    config = yaml.load(f, Loader=yaml.FullLoader)
    config = Config(config)
    return config
