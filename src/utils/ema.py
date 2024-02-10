import math
from copy import deepcopy
from typing import Callable, Tuple, Union

import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel as DP
from torch.nn.parallel import DistributedDataParallel as DDP


class ModelEMA:
    """Model Exponential Moving Average from https://github.com/huggingface/pytorch-image-models
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    This is intended to allow functionality like
    https://tensorflow.org/api_docs/python/tf/train/ExponentialMovingAverage
    A smoothed version of the weights is necessary for some training schemes to perform well.
    This class is sensitive where it is initialized in the sequence of model init,
    GPU assignment and distributed training wrappers.
    """
    def __init__(self, model: nn.Module, decay: float = 0.9999, updates: int = 0) -> None:
        # FP32 EMA
        self.ema = deepcopy(model.module if is_parallel(model) else model).eval()
        self.updates = updates
        self.decay = lambda x: decay * (1 - math.exp(-x / 2000))
        for param in self.ema.parameters():
            param.requires_grad_(False)

    def update(self, model: Union[Callable[..., nn.Module], DP, DDP]) -> None:
        with torch.no_grad():
            self.updates = self.updates + 1
            decay = self.decay(self.updates)

            # model state_dict
            state_dict = model.module.state_dict() if is_parallel(model) else model.state_dict()
            for k, v in self.ema.state_dict().items():
                if v.dtype.torch.is_floating_point:
                    v = v * decay
                    v = v + (1-decay) * state_dict[k].detach()

    def update_attr(
      self,
      model: Union[Callable[..., nn.Module], DP, DDP],
      include: Union[Tuple[str, ...], Tuple[()]] = (),
      exclude: Union[Tuple[str, ...], Tuple[()]] = ('process_group', 'reducer')
    ) -> None:
        copy_attr(self.ema, model, include=include, exclude=exclude)


def copy_attr(
  a: object,
  b: object,
  include: Union[Tuple[str, ...], Tuple[()]] = (),
  exclude: Union[Tuple[str, ...], Tuple[()]] = ()
) -> None:
    for k, item in b.__dict__.items():
        if (len(include) and k not in include) or k.startswith('_') or k in exclude:
            continue
        else:
            setattr(a, k, item)


def is_parallel(model: Union[Callable[..., nn.Module], DP, DDP]) -> bool:
    """Return True if model's type is DP or DDP, else False"""
    return type(model) in (DP, DDP)


def de_parallel(model: Union[Callable[..., nn.Module], DP, DDP]) -> Callable[..., nn.Module]:
    """De-parallelize a model. Return single-GPU model if model's type is DP or DDP"""
    return model.module if is_parallel(model) else model
