from contextlib import contextmanager
from typing import Generator

import torch.distributed as dist
import torch.nn as nn


@contextmanager
def torch_distributed_zero_first(local_rank: int) -> Generator:
    """a context manager that ensures only the first process in a distributed training setup

    :param local_rank: (int) the rank of the current process
    """
    if local_rank not in (-1, 0):
        dist.barrier(device_ids=[local_rank])
    yield
    if local_rank == 0:  # master process
        dist.barrier(device_ids=[0])


def initialize_weights(model: nn.Module) -> None:
    """initializes the weights of a PyTorch model

    :param model: (module) the model to initialize the weights for
    """
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            pass
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.LeakyReLU, nn.Hardswish)):
            m.inplace = True
