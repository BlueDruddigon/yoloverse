from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.layers.common import CSPSPPF, SPPF, BepC3, ConvBNSiLU, RepBlock, RepVGGBlock, SimCSPSPPF, SimSPPF


class EfficientRep(nn.Module):
    """EfficientRep Backbone.
    EfficientRep is handcrafted by hardware-aware neural network design.
    With Rep-Style structure, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """

    num_stages: int = 4

    def __init__(
      self,
      in_channels: int = 3,
      channels_list: Optional[List[int]] = None,
      num_repeats: Optional[List[int]] = None,
      block: Callable[..., nn.Module] = RepVGGBlock,
      fuse_p2: bool = False,
      cspsppf: bool = False
    ) -> None:
        """Initializes the EfficientRep backbone.

        :param in_channels: number of input channels. default: 3
        :param channels_list: list of channel sizes for each stage of the backbone. default: None
        :param num_repeats: list of number of repetitions for each stage of the backbone. default: None
        :param block: block to use in the backbone. default: RepVGGBlock
        :param fuse_p2: whether to fuse P2 feature map with the backbone output. default: False
        :param cspsppf: whether to use CSPSPPF or SimCSPSPPF for channel merging. default: False
        """
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_p2 = fuse_p2

        self.stem = block(in_channels=in_channels, out_channels=channels_list[0], kernel_size=3, stride=2)
        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            layer = nn.Sequential(
              block(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=2),
              RepBlock(
                in_channels=channels_list[i + 1],
                out_channels=channels_list[i + 1],
                num_block=num_repeats[i + 1],
                block=block
              )
            ) if i < self.num_stages - 1 else nn.Sequential(
              block(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=2),
              RepBlock(
                in_channels=channels_list[i + 1],
                out_channels=channels_list[i + 1],
                num_block=num_repeats[i + 1],
                block=block
              ),
              channel_merge_layer(in_channels=channels_list[i + 1], out_channels=channels_list[i + 1], kernel_size=5),
            )
            self.blocks.append(layer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs: List[torch.Tensor] = []
        x = self.stem(x)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx == 0 and not self.fuse_p2:
                continue
            outputs.append(x)
        return tuple(outputs)


class EfficientRep6(nn.Module):
    """EfficientRep+P6 Backbone.
    EfficientRep is handcrafted by hardware-aware neural network design.
    With Rep-Style structure, EfficientRep is friendly to high-computation hardware(e.g. GPU).
    """
    num_stages: int = 5

    def __init__(
      self,
      in_channels: int = 3,
      channels_list: Optional[List[int]] = None,
      num_repeats: Optional[List[int]] = None,
      block: Callable[..., nn.Module] = RepVGGBlock,
      fuse_p2: bool = False,
      cspsppf: bool = False
    ) -> None:
        """Initializes the EfficientRep6 backbone.

        :param in_channels: number of input channels. default: 3
        :param channels_list: list of channel sizes for each stage of the backbone. default: None
        :param num_repeats: list of number of repetitions for each stage of the backbone. default: None
        :param block: block to use in the backbone. default: RepVGGBlock
        :param fuse_p2: whether to fuse P2 feature map with the backbone output. default: False
        :param cspsppf: whether to use CSPSPPF or SimCSPSPPF for channel merging. default: False
        """
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_p2 = fuse_p2

        self.stem = block(in_channels=in_channels, out_channels=channels_list[0], kernel_size=3, stride=2)
        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            layer = nn.Sequential(
              block(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=2),
              RepBlock(
                in_channels=channels_list[i + 1],
                out_channels=channels_list[i + 1],
                num_block=num_repeats[i + 1],
                block=block
              )
            ) if i < self.num_stages - 1 else nn.Sequential(
              block(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=2),
              RepBlock(
                in_channels=channels_list[i + 1],
                out_channels=channels_list[i + 1],
                num_block=num_repeats[i + 1],
                block=block
              ),
              channel_merge_layer(in_channels=channels_list[i + 1], out_channels=channels_list[i + 1], kernel_size=5),
            )
            self.blocks.append(layer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs: List[torch.Tensor] = []
        x = self.stem(x)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx == 0 and not self.fuse_p2:
                continue
            outputs.append(x)
        return tuple(outputs)


class CSPBepBackbone(nn.Module):
    """
    CSPBepBackbone Module.
    It's composed by BepC3 Unit instead of RepBlock in EfficientRep.
    """
    num_stages: int = 4

    def __init__(
      self,
      in_channels: int = 3,
      channels_list: Optional[List[int]] = None,
      num_repeats: Optional[List[int]] = None,
      block: Callable[..., nn.Module] = RepVGGBlock,
      csp_ratio: float = 1. / 2,
      fuse_p2: bool = False,
      cspsppf: bool = False,
      stage_block_type: str = 'BepC3'
    ) -> None:
        """Initializes the CSPBep backbone.

        :param in_channels: number of input channels. default: 3
        :param channels_list: list of channel sizes for each stage. default: None
        :param num_repeats: list of block quantities to repeat for each stage. default: None
        :param block: block type to use in the backbone. default: RepVGGBlock
        :param csp_ratio: ratio of hidden channels in the CSP block. default: 0.5
        :param fuse_p2: whether to fuse P2 feature map. default: False
        :param cspsppf: whether to use CSPSPPF block instead of SPPF block. default: False
        :param stage_block_type: type of block to use in the stage. default: 'BepC3'
        """
        super().__init__()

        assert channels_list is not None
        assert num_repeats is not None
        self.fuse_p2 = fuse_p2

        if stage_block_type == 'BepC3':
            stage_block = BepC3
        else:
            raise NotImplementedError

        self.stem = block(in_channels=in_channels, out_channels=channels_list[0], kernel_size=3, stride=2)
        channel_merge_layer = SPPF if block == ConvBNSiLU else SimSPPF
        if cspsppf:
            channel_merge_layer = CSPSPPF if block == ConvBNSiLU else SimCSPSPPF

        self.blocks = nn.ModuleList()
        for i in range(self.num_stages):
            layer = nn.Sequential(
              block(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=2),
              stage_block(
                in_channels=channels_list[i + 1],
                out_channels=channels_list[i + 1],
                num_block=num_repeats[i + 1],
                hidden_ratio=csp_ratio,
                block=block
              )
            ) if i < self.num_stages - 1 else nn.Sequential(
              block(in_channels=channels_list[i], out_channels=channels_list[i + 1], kernel_size=3, stride=2),
              stage_block(
                in_channels=channels_list[i + 1],
                out_channels=channels_list[i + 1],
                num_block=num_repeats[i + 1],
                hidden_ratio=csp_ratio,
                block=block
              ),
              channel_merge_layer(in_channels=channels_list[i + 1], out_channels=channels_list[i + 1], kernel_size=5)
            )
            self.blocks.append(layer)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        outputs: List[torch.Tensor] = []
        x = self.stem(x)
        for idx, blk in enumerate(self.blocks):
            x = blk(x)
            if idx == 0 and not self.fuse_p2:
                continue
            outputs.append(x)
        return tuple(outputs)
