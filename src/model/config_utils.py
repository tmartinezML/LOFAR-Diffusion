from inspect import signature
from typing import Any


import torch.nn as nn


class modelConfig(object):
    def __init__(self, **kwargs):
        self.param_dict = kwargs
        self.__dict__.update(self.param_dict)

    def __setattr__(self, __name: str, __value: Any) -> None:
        super.__setattr__(self, __name, __value)
        if __name != "param_dict":
            self.param_dict[__name] = __value

    def update(self, update_dict):
        self.param_dict.update(update_dict)
        self.__dict__.update(self.param_dict)


def construct_from_config(cls, config, *args, **kwargs):
    # Extract valid kwargs from hyperparams
    config_kwargs = {
        k: v for k, v in config.param_dict.items()
        if k in signature(cls).parameters.keys()
    }
    return cls(*args, **(kwargs | config_kwargs))


class configModuleBase(nn.Module):
    @classmethod
    def from_config(cls, config, *args, **kwargs):

        # Special case for Unet
        if (
            hasattr(config, 'context')
            and 'context_dim' in signature(cls).parameters.keys()
        ):
            config.context_dim = len(config.context)

        return construct_from_config(
            cls, config, *args, **kwargs
        )


def isModel(model, modelClass):
    if isinstance(model, nn.DataParallel):
        return isinstance(model.module, modelClass)
    return isinstance(model, modelClass)
