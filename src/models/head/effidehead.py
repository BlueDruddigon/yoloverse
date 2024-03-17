import math
from typing import List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.assigners.anchor_generator import generate_anchors
from src.layers.common import ConvBNSiLU
from src.utils.bbox import dist2bbox


class EffiDeHead(nn.Module):
    def __init__(
      self,
      num_classes: int = 80,
      anchors: int | Sequence[Sequence[int]] = 1,
      num_layers: int = 3,
      inplace: bool = True,
      head_layers: Optional[nn.Sequential] = None,
      use_dfl: bool = True,
      reg_max: int = 16
    ) -> None:
        super().__init__()
        assert head_layers is not None

        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers

        # number of anchor points
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors

        # grid cells and anchor points
        self.anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        strides = [8, 16, 32]  # feature maps' strides computed during build
        self.strides = torch.tensor(strides)

        # hyper-parameters
        self.prior_prob = 1e-2
        self.inplace = inplace
        self.use_dfl = use_dfl
        self.reg_max = reg_max

        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, kernel_size=1, bias=False)

        # init decouple heads
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # efficient decoupled head layers
        for i in range(num_layers):
            idx = i * 5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx + 1])
            self.reg_convs.append(head_layers[idx + 2])
            self.cls_preds.append(head_layers[idx + 3])
            self.reg_preds.append(head_layers[idx + 4])

    def initialize_biases(self) -> None:
        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = nn.Parameter(b.view(-1, ), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.)
            conv.bias = nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = nn.Parameter(w, requires_grad=True)

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(
          self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(), requires_grad=False
        )

    def forward(self, x: List[Tensor]) -> Tuple[Sequence[Tensor], Tensor, Tensor] | Tensor:
        if self.training:
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute(0, 2, 1))
                reg_dist_list.append(reg_output.flatten(2).permute(0, 2, 1))

            cls_score_list = torch.cat(cls_score_list, dim=1)
            reg_dist_list = torch.cat(reg_dist_list, dim=1)

            return x, cls_score_list, reg_dist_list

        # eval mode
        cls_score_list = []
        reg_dist_list = []

        # generate anchor points
        anchor_points, stride_tensor = generate_anchors(
          x,
          self.strides,
          grid_cell_size=self.grid_cell_size,
          grid_cell_offset=self.grid_cell_offset,
          device=x[0].device,
          is_eval=True
        )

        for i in range(self.nl):
            B, _, H, W = x[i].shape
            L = H * W
            x[i] = self.stems[i](x[i])
            cls_x = x[i]
            reg_x = x[i]

            cls_feat = self.cls_convs[i](cls_x)
            cls_output = self.cls_preds[i](cls_feat)
            reg_feat = self.reg_convs[i](reg_x)
            reg_output = self.reg_preds[i](reg_feat)

            if self.use_dfl:
                reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, L]).permute(0, 2, 1, 3)
                reg_output = self.proj_conv(F.softmax(reg_output, dim=1))

            cls_output = torch.sigmoid(cls_output)
            cls_score_list.append(cls_output.reshape([B, self.nc, L]))
            reg_dist_list.append(reg_output.reshape([B, 4, L]))

        cls_score_list = torch.cat(cls_score_list, dim=-1).permute(0, 2, 1)
        reg_dist_list = torch.cat(reg_dist_list, dim=-1).permute(0, 2, 1)

        pred_bboxes = dist2bbox(reg_dist_list, anchor_points=anchor_points, bbox_format='xywh')
        pred_bboxes = pred_bboxes * stride_tensor

        return torch.cat(
          [
            pred_bboxes,
            torch.ones([B, pred_bboxes.shape[1], 1], device=pred_bboxes.device, dtype=pred_bboxes.dtype),
            cls_score_list,
          ],
          dim=-1,
        )


def build_effidehead_layers(
  channels_list: Optional[Sequence[int]],
  num_anchors: int,
  num_classes: int,
  reg_max: int = 16,
  num_layers: int = 3
) -> nn.Sequential:
    assert channels_list is not None
    assert num_layers == 3

    channels_out = [6, 8, 10]  # output indices from the neck

    head_layers = nn.Sequential(
      # stem0
      ConvBNSiLU(
        in_channels=channels_list[channels_out[0]],
        out_channels=channels_list[channels_out[0]],
        kernel_size=1,
        stride=1
      ),
      # cls_conv0
      ConvBNSiLU(in_channels=channels_list[channels_out[0]], out_channels=channels_list[channels_out[0]]),
      # reg_conv0
      ConvBNSiLU(in_channels=channels_list[channels_out[0]], out_channels=channels_list[channels_out[0]]),
      # cls_pred0
      nn.Conv2d(in_channels=channels_list[channels_out[0]], out_channels=num_classes * num_anchors, kernel_size=1),
      # reg_pred0
      nn.Conv2d(in_channels=channels_list[channels_out[0]], out_channels=4 * (reg_max+num_anchors), kernel_size=1),
      # stem1
      ConvBNSiLU(
        in_channels=channels_list[channels_out[1]],
        out_channels=channels_list[channels_out[1]],
        kernel_size=1,
        stride=1
      ),
      # cls_conv1
      ConvBNSiLU(in_channels=channels_list[channels_out[1]], out_channels=channels_list[channels_out[1]]),
      # reg_conv1
      ConvBNSiLU(in_channels=channels_list[channels_out[1]], out_channels=channels_list[channels_out[1]]),
      # cls_pred1
      nn.Conv2d(in_channels=channels_list[channels_out[1]], out_channels=num_classes * num_anchors, kernel_size=1),
      # reg_pred1
      nn.Conv2d(in_channels=channels_list[channels_out[1]], out_channels=4 * (reg_max+num_anchors), kernel_size=1),
      # stem2
      ConvBNSiLU(
        in_channels=channels_list[channels_out[2]],
        out_channels=channels_list[channels_out[2]],
        kernel_size=1,
        stride=1
      ),
      # cls_conv2
      ConvBNSiLU(in_channels=channels_list[channels_out[2]], out_channels=channels_list[channels_out[2]]),
      # reg_conv2
      ConvBNSiLU(in_channels=channels_list[channels_out[2]], out_channels=channels_list[channels_out[2]]),
      # cls_pred2
      nn.Conv2d(in_channels=channels_list[channels_out[2]], out_channels=num_classes * num_anchors, kernel_size=1),
      # reg_pred2
      nn.Conv2d(in_channels=channels_list[channels_out[2]], out_channels=4 * (reg_max+num_anchors), kernel_size=1),
    )
    return head_layers
