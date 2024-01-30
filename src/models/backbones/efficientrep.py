from typing import Callable, List, Optional, Tuple

import torch
import torch.nn as nn

from src.layers.common import CSPSPPF, SPPF, ConvBNSiLU, RepBlock, RepVGGBlock, SimCSPSPPF, SimSPPF


class EfficientRep(nn.Module):
    """EfficientRep Backbone
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
