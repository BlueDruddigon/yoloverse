import math
from typing import Any, Callable, Tuple

import torch.nn as nn
import torch.optim as optim

from src.utils.config import Config
from src.utils.logger import LOGGER


def build_optimizer(cfg: Config, model: nn.Module) -> optim.Optimizer:
    """build optimizer from cfg file.

    :param cfg: (Config) config dict
    :param model: (nn.Module) model to be optimized
    :return: (optim.Optimizer) optimizer to be used
    """
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    if cfg.solver.optim == 'SGD':
        optimizer = optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    del g_bnw, g_w, g_b
    return optimizer


def build_lr_scheduler(
  cfg: Config,
  optimizer: optim.Optimizer,
  epochs: int,
) -> Tuple[optim.lr_scheduler.LambdaLR, Callable[..., Any]]:
    """build learning rate scheduler from config file

    :param cfg: (Config) config dict
    :param optimizer: (optim.Optimizer) optimizer to used
    :param epochs: (int) number of epochs
    :return: a tuple contains: a LambdaLR instance and a callable of decay rules
    """
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda _: 1.0
    else:
        LOGGER.error('unknown lr scheduler, use Cosine defaulted')
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)
    return scheduler, lf
