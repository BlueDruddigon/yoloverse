# we'll implement Rep-PAN module for YOLOv6 in this video
from typing import Callable, Optional, Sequence

import torch
import torch.nn as nn

from src.layers.common import ConvBNReLU, RepBlock, RepVGGBlock, Transpose


class RepPANNeck(nn.Module):
    def __init__(
      self,
      channels_list: Optional[Sequence[int]] = None,
      num_repeats: Optional[Sequence[int]] = None,
      block: Callable[..., nn.Module] = RepVGGBlock
    ) -> None:
        super().__init__()
        
        assert channels_list is not None
        assert num_repeats is not None
        
        self.reduce_layer0 = ConvBNReLU(
          in_channels=channels_list[4], out_channels=channels_list[5], kernel_size=1, stride=1
        )
        
        self.upsample0 = Transpose(in_channels=channels_list[5], out_channels=channels_list[5])
        
        self.Rep_p4 = RepBlock(
          in_channels=channels_list[3] + channels_list[5],
          out_channels=channels_list[5],
          num_block=num_repeats[5],
          block=block
        )
        
        self.reduce_layer1 = ConvBNReLU(
          in_channels=channels_list[5], out_channels=channels_list[6], kernel_size=1, stride=1
        )
        
        self.upsample1 = Transpose(in_channels=channels_list[6], out_channels=channels_list[6])
        
        self.Rep_p3 = RepBlock(
          in_channels=channels_list[2] + channels_list[6],
          out_channels=channels_list[6],
          num_block=num_repeats[6],
          block=block
        )
        
        self.downsample1 = ConvBNReLU(
          in_channels=channels_list[6], out_channels=channels_list[7], kernel_size=3, stride=2
        )
        
        self.Rep_n3 = RepBlock(
          in_channels=channels_list[6] + channels_list[7],
          out_channels=channels_list[8],
          num_block=num_repeats[7],
          block=block
        )
        
        self.downsample0 = ConvBNReLU(
          in_channels=channels_list[8], out_channels=channels_list[9], kernel_size=3, stride=2
        )
        
        self.Rep_n4 = RepBlock(
          in_channels=channels_list[5] + channels_list[9],
          out_channels=channels_list[10],
          num_block=num_repeats[8],
          block=block
        )
    
    def forward(self, inputs: Sequence[torch.Tensor]) -> Sequence[torch.Tensor]:
        (x2, x1, x0) = inputs
        
        fpn_out0 = self.reduce_layer0(x0)
        upsample_feat0 = self.upsample0(fpn_out0)
        f_concat_layer0 = torch.cat([x1, upsample_feat0], dim=1)
        feat_out0 = self.Rep_p4(f_concat_layer0)
        
        fpn_out1 = self.reduce_layer1(feat_out0)
        upsample_feat1 = self.upsample1(fpn_out1)
        f_concat_layer1 = torch.cat([x2, upsample_feat1], dim=1)
        pan_out2 = self.Rep_p3(f_concat_layer1)
        
        down_feat1 = self.downsample1(pan_out2)
        p_concat_layer1 = torch.cat([fpn_out1, down_feat1], dim=1)
        pan_out1 = self.Rep_n3(p_concat_layer1)
        
        down_feat0 = self.downsample0(pan_out1)
        p_concat_layer0 = torch.cat([fpn_out0, down_feat0], dim=1)
        pan_out0 = self.Rep_n4(p_concat_layer0)
        
        outputs = [pan_out2, pan_out1, pan_out0]
        return outputs
