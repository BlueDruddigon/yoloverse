import torch.nn as nn


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
