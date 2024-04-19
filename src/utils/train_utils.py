# Decorator class
from copy import deepcopy

import torch


class use_ema:
    def __init__(self, model, ema_model):
        self.model = model
        self.ema_model = ema_model

    def __enter__(self):
        self.model_state = deepcopy(self.model.state_dict())
        self.model.load_state_dict(self.ema_model.module.state_dict())

    def __exit__(self, *args):
        self.model.load_state_dict(self.model_state)


def get_power_ema_avg_fn(gamma):
    @torch.no_grad()
    def ema_update(ema_param: torch.Tensor, current_param: torch.Tensor, num_averaged):
        beta = (1 - 1 / num_averaged)**(gamma + 1)
        return beta * ema_param + (1 - beta) * current_param

    return ema_update
