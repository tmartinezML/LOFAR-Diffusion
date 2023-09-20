from inspect import signature

import torch.nn as nn

def construct_from_config(cls, config):
    # Extract valid kwargs from hyperparams
        kwargs = {
            k: v for k, v in config.param_dict.items()
            if k in signature(cls).parameters.keys()
        }
        return cls(**kwargs)

class modelConfig(object):
    def __init__(self, **kwargs):
        self.param_dict = kwargs
        self.__dict__.update(kwargs)
    
class customModelClass(nn.Module):
    @classmethod
    def from_config(cls, config):
        return construct_from_config(cls, config)

