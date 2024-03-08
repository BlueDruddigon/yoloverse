from contextlib import contextmanager
from typing import Generator

import torch
import torch.distributed as dist
import torch.nn as nn

from src.layers.common import ConvModule


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


def fuse_conv_bn(conv: nn.Conv2d, bn: nn.BatchNorm2d) -> nn.Conv2d:
    """fuse a Conv2D layer with a BatchNorm2D layer

    :param conv: (nn.Conv2d) the convolutional layer
    :param bn: (nn.BatchNorm2d) the batch normalization layer
    :returns: (nn.Conv2d) the fused convolutional layer
    :raises: AssertionError if the layer's validity is not ensured
    """
    # ensure layer's validity
    assert isinstance(conv.kernel_size, tuple) and len(conv.kernel_size) == 2
    assert isinstance(conv.stride, tuple) and len(conv.stride) == 2
    assert isinstance(conv.padding, tuple) and len(conv.padding) == 2
    assert bn.running_mean is not None and bn.running_var is not None

    # create a new conv layer with the same parameters as the original one
    fused_conv = nn.Conv2d(
      conv.in_channels,
      conv.out_channels,
      kernel_size=conv.kernel_size,
      stride=conv.stride,
      padding=conv.padding,
      groups=conv.groups,
      bias=True
    ).requires_grad_(False).to(conv.weight.device)

    # prepare filters for the new conv layer by fusing the weights of the original conv and bn layers
    w_conv = conv.weight.clone().view(conv.out_channels, -1)
    w_bn = torch.diag(bn.weight.div(torch.sqrt(bn.eps + bn.running_var)))
    fused_conv.weight.copy_(torch.mm(w_bn, w_conv).view(fused_conv.weight.shape))

    # prepare spatial bias for the new conv layer by fusing the bias of the original conv and bn layers
    b_conv = torch.zeros(conv.weight.size(0), device=conv.weight.device) if conv.bias is None else conv.bias
    b_bn = bn.bias - bn.weight.mul(bn.running_mean).div(torch.sqrt(bn.running_var + bn.eps))
    fused_conv.bias.copy_(torch.mm(w_bn, b_conv.reshape(-1, 1)).reshape(-1) + b_bn)

    return fused_conv


def fuse_model(model: nn.Module) -> nn.Module:
    """fuse the Conv2D layers with BatchNorm2D layers in the given model's ConvModule

    :param model: (nn.Module) the original model to fuse
    :returns: (nn.Module) the fused model
    """
    for m in model.modules():
        # check if the module is a ConvModule and has a batch norm layer
        if isinstance(m, ConvModule) and hasattr(m, 'bn'):
            m.conv = fuse_conv_bn(m.conv, m.bn)  # fusing conv with bn
            delattr(m, 'bn')  # remove unused bn
            m.forward = m.forward_fuse  # update forward method to use the fused version
    return model
