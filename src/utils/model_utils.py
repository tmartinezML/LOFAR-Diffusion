from inspect import signature

import torch.nn as nn


def construct_from_config(cls, config, *args, **kwargs):
    # Extract valid kwargs from hyperparams
        config_kwargs = {
            k: v for k, v in config.param_dict.items()
            if k in signature(cls).parameters.keys()
        }
        return cls(*args, **(kwargs | config_kwargs))

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
        
        self.__dict__.update(self.param_dict)
    
class customModelClass(nn.Module):
    @classmethod
    def from_config(cls, config):
        return construct_from_config(cls, config)

def isModel(model, modelClass):
    if isinstance(model, nn.DataParallel):
        return isinstance(model.module, modelClass)
    return isinstance(model, modelClass)

