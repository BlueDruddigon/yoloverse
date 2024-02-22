import torch.nn as nn


def initialize_weights(model: nn.Module) -> None:
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            pass
        elif isinstance(m, nn.BatchNorm2d):
            m.eps = 1e-3
            m.momentum = 0.03
        elif isinstance(m, (nn.ReLU, nn.ReLU6, nn.SiLU, nn.LeakyReLU, nn.Hardswish)):
            m.inplace = True
