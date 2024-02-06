import warnings
from typing import Callable, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# activation function mappings table
act_table = {
  'relu': nn.ReLU(),
  'silu': nn.SiLU(),
  'hardswish': nn.Hardswish(),
}


class ConvModule(nn.Module):
    """A combination of Conv + BN + Activation"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      stride: int,
      act_type: Optional[str],
      padding: Optional[int] = None,
      groups: int = 1,
      bias: bool = False
    ) -> None:
        """Initializes of ConvModule.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: conv's kernel size.
        :param stride: conv's stride size.
        :param act_type: activation type to use.
        :param padding: zeros-padding size. default: None
        :param groups: conv's groups. default: 1
        :param bias: whether to use conv's bias. default: False
        """
        super().__init__()
        if padding is None:
            padding = kernel_size // 2
        self.conv = nn.Conv2d(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=kernel_size,
          stride=stride,
          padding=padding,
          groups=groups,
          bias=bias
        )
        self.bn = nn.BatchNorm2d(out_channels)
        if act_type is not None and act_type in act_table:
            self.act = act_table[act_type]
        else:
            self.act = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.conv(x))


class ConvBNReLU(nn.Module):
    """Conv + BN + ReLU Module"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      padding: Optional[int] = None,
      groups: int = 1,
      bias: bool = False
    ) -> None:
        """Initializes of ConvBNReLU.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: conv's kernel size. default: 3
        :param stride: conv's stride size. default: 1
        :param padding: zeros-padding size. default: None
        :param groups: conv's groups. default: 1
        :param bias: whether to use conv's bias. default: False
        """
        super().__init__()
        self.block = ConvModule(
          in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          act_type='relu',
          padding=padding,
          groups=groups,
          bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBNSiLU(nn.Module):
    """Conv + BN + SiLU Module"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      padding: Optional[int] = None,
      groups: int = 1,
      bias: bool = False
    ) -> None:
        """Initializes of ConvBNSiLU.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: conv's kernel size. default: 3
        :param stride: conv's stride size. default: 1
        :param padding: zeros-padding size. default: None
        :param groups: conv's groups. default: 1
        :param bias: whether to use conv's bias. default: False
        """
        super().__init__()
        self.block = ConvModule(
          in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          act_type='silu',
          padding=padding,
          groups=groups,
          bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBNHS(nn.Module):
    """Conv + BN + Hasrdswish Module"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      padding: Optional[int] = None,
      groups: int = 1,
      bias: bool = False
    ) -> None:
        """Initializes of ConvBNHS.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: conv's kernel size. default: 3
        :param stride: conv's stride size. default: 1
        :param padding: zeros-padding size. default: None
        :param groups: conv's groups. default: 1
        :param bias: whether to use conv's bias. default: False
        """
        super().__init__()
        self.block = ConvModule(
          in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          act_type='hardswish',
          padding=padding,
          groups=groups,
          bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class ConvBN(nn.Module):
    """Conv + BN Module without activation"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      padding: Optional[int] = None,
      groups: int = 1,
      bias: bool = False
    ) -> None:
        """Initializes of ConvBN.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: conv's kernel size. default: 3
        :param stride: conv's stride size. default: 1
        :param padding: zeros-padding size. default: None
        :param groups: conv's groups. default: 1
        :param bias: whether to use conv's bias. default: False
        """
        super().__init__()
        self.block = ConvModule(
          in_channels,
          out_channels,
          kernel_size=kernel_size,
          stride=stride,
          act_type=None,
          padding=padding,
          groups=groups,
          bias=bias
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class SPPFModule(nn.Module):
    """
    Spatial Pyramid Pooling - Fast Module
    Source code for SPPF refers to (taken from YOLOv5):
        https://github.com/ultralytics/yolov5/blob/4878541d43abce1ad7c670b7b8885e5dc8ddaeef/models/common.py#L242
    """
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 5,
      block: Callable[..., nn.Module] = ConvBNReLU
    ) -> None:
        """Initializes the SPPFModule.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: pooling layer's kernel size. default: 5
        :param block: block to use in this module. default: ConvBNReLU
        """
        super().__init__()
        hidden_channels = in_channels // 2
        self.cv1 = block(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = block(hidden_channels * 4, out_channels, kernel_size=1, stride=1)
        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], dim=1))


class SimSPPF(nn.Module):
    """Simplified SPPF with ReLU activation"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 5,
      block: Callable[..., nn.Module] = ConvBNReLU
    ) -> None:
        """Initializes the SimSPPF.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: pooling layer's kernel size. default: 5
        :param block: block to use in this module. default: ConvBNReLU
        """
        super().__init__()
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size=kernel_size, block=block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sppf(x)


class SPPF(nn.Module):
    """SPPF module with SiLU activation"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 5,
      block: Callable[..., nn.Module] = ConvBNSiLU
    ) -> None:
        """Initializes the SPPF.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: pooling layer's kernel size. default: 5
        :param block: block to use in this module. default: ConvBNSiLU
        """
        super().__init__()
        self.sppf = SPPFModule(in_channels, out_channels, kernel_size=kernel_size, block=block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sppf(x)


class CSPSPPFModule(nn.Module):
    """
    Cross Stage Partial Spatial Pyramid Pooling - Fast Module
    Source code for CSP refers to: https://github.com/WongKinYiu/CrossStagePartialNetworks
    """
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int,
      hidden_ratio: float = 0.5,
      block: Callable[..., nn.Module] = ConvBNReLU
    ) -> None:
        """Initializes the CSPSPPFModule.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: pooling layer's kernel size.
        :param hidden_ratio: ratio for hidden channels in CSP. default: 0.5
        :param block: block to use in this module. default: ConvBNReLU
        """
        super().__init__()
        hidden_channels = int(in_channels * hidden_ratio)
        self.cv1 = block(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = block(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv3 = block(hidden_channels, hidden_channels, kernel_size=3, stride=1)
        self.cv4 = block(hidden_channels, hidden_channels, kernel_size=1, stride=1)

        self.m = nn.MaxPool2d(kernel_size=kernel_size, stride=1, padding=kernel_size // 2)
        self.cv5 = block(4 * hidden_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv6 = block(hidden_channels, hidden_channels, kernel_size=3, stride=1)
        self.cv7 = block(2 * hidden_channels, out_channels, kernel_size=1, stride=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y0 = self.cv2(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            y1 = self.m(x1)
            y2 = self.m(y1)
            y3 = self.cv6(self.cv5(torch.cat([x1, y1, y2, self.m(y2)], dim=1)))
        return self.cv7(torch.cat([y0, y3], dim=1))


class SimCSPSPPF(nn.Module):
    """Simplied CSPSPPF module with ReLU activation"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 5,
      hidden_ratio: float = 0.5,
      block: Callable[..., nn.Module] = ConvBNReLU
    ) -> None:
        """Initializes the SimCSPSPPF.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: pooling layer's kernel size. default: 5
        :param hidden_ratio: ratio for hidden channels in CSP. default: 0.5
        :param block: block to use in this module. default: ConvBNReLU
        """
        super().__init__()
        self.cspsppf = CSPSPPFModule(
          in_channels, out_channels, kernel_size=kernel_size, hidden_ratio=hidden_ratio, block=block
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cspsppf(x)


class CSPSPPF(nn.Module):
    """CSPSPPF module with SiLU activation"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 5,
      hidden_ratio: float = 0.5,
      block: Callable[..., nn.Module] = ConvBNSiLU
    ) -> None:
        """Initializes the CSPSPPF.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: pooling layer's kernel size. default: 5
        :param hidden_ratio: ratio for hidden channels in CSP. default: 0.5
        :param block: block to use in this module. default: ConvBNSiLU
        """
        super().__init__()
        self.cspsppf = CSPSPPFModule(
          in_channels, out_channels, kernel_size=kernel_size, hidden_ratio=hidden_ratio, block=block
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cspsppf(x)


class Transpose(nn.Module):
    """Normal Transpose, default for upsampling"""
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 2, stride: int = 2) -> None:
        """Initializes the Transpose Layer.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param kernel_size: kernel size for `nn.ConvTranspose2d`. default: 2
        :param stride: stride for `nn.ConvTranspose2d`. default: 2
        """
        super().__init__()
        self.upsample_conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.upsample_conv(x)


class RepVGGBlock(nn.Module):
    """
    RepVGGBlock is a basic Rep-Style block, including training and deploy status
    This code is based on https://github.com/DingXiaoH/RepVGG/blob/main/repvgg.py
    """
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      kernel_size: int = 3,
      stride: int = 1,
      padding: int = 1,
      dilation: int = 1,
      groups: int = 1,
      deploy: bool = False,
      use_se: bool = False,
      use_act: bool = True
    ) -> None:
        """RepVGGBlock building block

        :param in_channels: number of channels in the input image
        :param out_channels: number of channels produced by the convolution
        :param kernel_size: kernel size of convolution. Default: 3
        :param stride: stride of convolution. Default: 1
        :param padding: zero-padding added to both sides of the input. Default: 1
        :param dilation: spacing between kernrel elements. Default: 1
        :param groups: number of blocked connections from input channels to output channels. Default: 1
        :param deploy: whether to initialize in deploy mode. Default: False
        :param use_se: whether to use SEBlock. Default: False
        :param use_act: whether to use ReLU activation. Default: True
        """
        super().__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

        if use_se:
            raise NotImplementedError
        else:
            self.se = nn.Identity()

        self.act = nn.ReLU() if use_act else nn.Identity()

        if deploy:
            self.rbr_reparam = nn.Conv2d(
              in_channels,
              out_channels,
              kernel_size=kernel_size,
              stride=stride,
              padding=padding,
              dilation=dilation,
              groups=groups,
              bias=True
            )
        else:
            self.rbr_identity = nn.BatchNorm2d(in_channels) if out_channels == in_channels and stride == 1 else None
            self.rbr_dense = ConvModule(
              in_channels,
              out_channels,
              kernel_size=kernel_size,
              stride=stride,
              act_type=None,
              padding=padding,
              groups=groups
            )
            self.rbr_1x1 = ConvModule(
              in_channels, out_channels, kernel_size=1, stride=stride, act_type=None, padding=0, groups=groups
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if hasattr(self, 'rbreparam'):
            return self.act(self.se(self.rbr_reparam(x)))

        identity_out = 0
        if self.rbr_identity is not None:
            identity_out = self.rbr_identity(x)

        return self.act(self.se(self.rbr_dense(x) + self.rbr_1x1(x) + identity_out))

    def get_equivalent_kernel_bias(self) -> Tuple[torch.Tensor, torch.Tensor]:
        kernel3x3, bias3x3 = self._fuse_bn_tensor(self.rbr_dense)
        kernel1x1, bias1x1 = self._fuse_bn_tensor(self.rbr_1x1)
        kernelid, biasid = self._fuse_bn_tensor(self.rbr_identity)
        return kernel3x3 + self._pad_1x1_to_3x3_tensor(kernel1x1) + kernelid, bias3x3 + bias1x1 + biasid

    def _avg_to_3x3_tensor(self, avgp: nn.AvgPool2d) -> torch.Tensor:
        channels = self.in_channels
        groups = self.groups
        kernel_size = avgp.kernel_size
        if isinstance(kernel_size, (tuple, list)):
            assert len(kernel_size) == 2
            assert kernel_size[0] == kernel_size[1]
            kernel_size = kernel_size[0]
        input_dim = channels // groups
        k = torch.zeros(channels, input_dim, kernel_size, kernel_size)
        k[np.arange(channels), np.tile(np.arange(input_dim), groups), :, :] = 1. / kernel_size ** 2
        return k

    def _pad_1x1_to_3x3_tensor(self, kernel1x1: torch.Tensor) -> torch.Tensor:
        """method to pad 1x1 kernel tensor to 3x3 kernel tensor"""
        if kernel1x1 is None:
            return torch.tensor(0)
        else:
            return F.pad(kernel1x1, [1, 1, 1, 1])

    def _fuse_bn_tensor(self, branch: Optional[Union[ConvModule, nn.BatchNorm2d]]) -> Tuple[torch.Tensor, torch.Tensor]:
        """method to fuse BatchNorm layer to Conv layer"""
        if branch is None:
            return torch.tensor(0), torch.tensor(0)
        if isinstance(branch, ConvModule):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        elif isinstance(branch, nn.BatchNorm2d):
            if not hasattr(self, 'id_tensor'):
                input_dim = self.in_channels // self.groups
                kernel_value = np.zeros((self.in_channels, input_dim, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % input_dim, 1, 1] = 1
                self.id_tensor = torch.from_numpy(kernel_value).to(device=branch.weight.device)
            kernel = self.id_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        assert running_var is not None
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean*gamma/std

    def reparameterize(self):
        """method to switch from training mode to deploy mode"""
        if hasattr(self, 'rbr_reparam'):
            return

        kernel, bias = self.get_equivalent_kernel_bias()
        self.rbr_reparam = nn.Conv2d(
          self.rbr_dense.conv.in_channels,
          self.rbr_dense.conv.out_channels,
          kernel_size=self.kernel_size,
          stride=self.stride,
          padding=self.padding,
          dilation=self.rpbr_dense.conv.dilation,
          groups=self.rbre_dense.conv.groups,
          bias=True
        )
        self.rbr_reparam.weight.data = kernel
        self.rbr_reparam.bias.data = bias

        for param in self.parameters():
            param.detach_()
        self.__delattr__('rbr_dense')
        self.__delattr__('rbr_1x1')
        if hasattr(self, 'rbr_identity'):
            self.__delattr__('rbr_identity')
        if hasattr(self, 'id_tensor'):
            self.__delattr__('id_tensor')

        self.deploy = True


class RepBlock(nn.Module):
    """RepBlock is a stage block with Rep-Style Basic Block"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_block: int = 1,
      block: Callable[..., nn.Module] = RepVGGBlock,
      basic_block: Callable[..., nn.Module] = RepVGGBlock
    ) -> None:
        """Initializes the RepBlock.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param num_block: number of repeated blocks. default: 1
        :param block: block to compose this module. default: RepVGGBlock
        :param basic_block: basic block to use in `block`. default: RepVGGBlock
        """
        super().__init__()
        self.conv1 = block(in_channels, out_channels)
        self.block = nn.Sequential(
          *[block(out_channels, out_channels) for _ in range(num_block - 1)]
        ) if num_block > 1 else None
        if block == BottleRep:
            self.conv1 = BottleRep(in_channels, out_channels, basic_block=basic_block, weight=True)
            self.block = nn.Sequential(
              *[block(out_channels, out_channels, basic_block=basic_block, weight=True) for _ in range(num_block - 1)]
            ) if num_block > 1 else None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv1(x)
        if self.block is not None:
            x = self.block(x)
        return x


class BottleRep(nn.Module):
    """Rep-Style Bottleneck Module"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      basic_block: Callable[..., nn.Module] = RepVGGBlock,
      weight: bool = False
    ) -> None:
        """Initializes the BottleRep module.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param basic_block: basic block to use in this module. default: RepVGGBlock
        :param weight: whether to add weighted parameters for residual connections. default: False
        """
        super().__init__()
        self.conv1 = basic_block(in_channels, out_channels)
        self.conv2 = basic_block(out_channels, out_channels)
        if in_channels == out_channels:
            self.shortcut = True
        else:
            self.shortcut = False
        if weight:
            self.alpha = nn.Parameter(torch.ones(1))
        else:
            self.alpha = 1.

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv2(self.conv1(x))
        return out + self.alpha * x if self.shortcut else out


class BepC3(nn.Module):
    """CSPStackRep Block"""
    def __init__(
      self,
      in_channels: int,
      out_channels: int,
      num_block: int = 1,
      hidden_ratio: float = 0.5,
      block: Callable[..., nn.Module] = RepVGGBlock
    ) -> None:
        """Initializes the BepC3 module.

        :param in_channels: module's input channels.
        :param out_channels: module's output channels.
        :param num_block: number of repeated blocks. default: 1
        :param hidden_ratio: ratio for hidden channels. default: 0.5
        :param block: block to use in this module. default: RepVGGBlock
        """
        super().__init__()
        hidden_channels = int(out_channels * hidden_ratio)
        self.cv1 = ConvBNReLU(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv2 = ConvBNReLU(in_channels, hidden_channels, kernel_size=1, stride=1)
        self.cv3 = ConvBNReLU(hidden_channels * 2, out_channels, kernel_size=1, stride=1)
        if block == ConvBNSiLU:
            self.cv1 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1)
            self.cv2 = ConvBNSiLU(in_channels, hidden_channels, kernel_size=1, stride=1)
            self.cv3 = ConvBNSiLU(hidden_channels * 2, out_channels, kernel_size=1, stride=1)
        self.m = RepBlock(hidden_channels, hidden_channels, num_block=num_block, block=BottleRep, basic_block=block)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cv3(torch.cat([self.cv1(x), self.m(self.cv2(x))], dim=1))
