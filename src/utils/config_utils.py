from inspect import signature
from typing import Any


import torch.nn as nn


class modelConfig(object):
    def __init__(self, **kwargs):
        self.param_dict = kwargs

        # Backwards compatibility:
        kk = self.param_dict.keys()
        # Changes 25.09.23:
        if 'use_improved_unet' in kk:
            self.param_dict["model_type"] = "ImprovedUnet"
        if "loss_type" not in kk and "learn_variance" in kk:
            self.param_dict["loss_type"] = (
                "hybrid" if self.param_dict["learn_variance"] else "huber"
            )
        # Change 13.10.23: Snapshotting & lr decay
        if "snapshot_interval" not in kk:
            self.param_dict["snapshot_interval"] = None

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
