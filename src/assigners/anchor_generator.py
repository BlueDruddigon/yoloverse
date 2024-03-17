"""Anchor Generation function for YOLO's head and Loss computation"""
from typing import List, Sequence, Tuple

import torch


def generate_anchors(
  feats: Sequence[torch.Tensor],
  fpn_strides: torch.Tensor | Sequence[int],
  grid_cell_size: float = 5.0,
  grid_cell_offset: float = 0.5,
  device: str | torch.device = 'cpu',
  is_eval: bool = False
) -> Tuple[torch.Tensor, torch.Tensor] | Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]:
    """Generate anchor points from fpn feature maps

    :param feats: FPN feature maps
    :param fpn_strides: feature maps' strides
    :param grid_cell_size: size of each grid cell. default: 5.0
    :param grid_cell_offset: offset of grid cell. default: 0.5
    :param device: device to use for computation. default: 'cpu'
    :param is_eval: whether to use in evaluation mode. default: False
    :return: a tuple of:
        - anchor points and stride tensors if `is_eval` is True
        - anchors tensor, anchor points tensor, a list contains number of anchors and stride tensor, otherwise
    """
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None

    if is_eval:
        for i, stride in enumerate(fpn_strides):
            stride = stride.item() if isinstance(stride, torch.Tensor) else int(stride)
            _, _, H, W = feats[i].shape
            shift_x = torch.arange(W, device=device) + grid_cell_offset
            shift_y = torch.arange(H, device=device) + grid_cell_offset
            shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
            anchor_point = torch.stack([shift_x, shift_y], dim=-1).to(torch.float)
            anchor_points.append(anchor_point.reshape([-1, 2]))
            stride_tensor.append(torch.full((H * W, 1), stride, dtype=torch.float, device=device))

        anchor_points = torch.cat(anchor_points)
        stride_tensor = torch.cat(stride_tensor)
        return anchor_points, stride_tensor

    for i, stride in enumerate(fpn_strides):
        stride = stride.item() if isinstance(stride, torch.Tensor) else int(stride)
        _, _, H, W = feats[i].shape
        cell_half_size = grid_cell_size * stride * 0.5
        shift_x = (torch.arange(W, device=device) + grid_cell_offset) * stride
        shift_y = (torch.arange(H, device=device) + grid_cell_offset) * stride
        shift_y, shift_x = torch.meshgrid(shift_y, shift_x, indexing='ij')
        anchor = torch.stack(
          [
            shift_x - cell_half_size,
            shift_y - cell_half_size,
            shift_x + cell_half_size,
            shift_y + cell_half_size,
          ],
          dim=-1,
        ).clone().to(feats[0].dtype)
        anchor_point = torch.stack([shift_x, shift_y], dim=-1).clone().to(feats[0].dtype)

        anchors.append(anchor.reshape([-1, 4]))
        anchor_points.append(anchor_point.reshape([-1, 2]))
        num_anchors_list.append(len(anchors[-1]))
        stride_tensor.append(torch.full([num_anchors_list[-1], 1], stride, dtype=feats[0].dtype))

    anchors = torch.cat(anchors)
    anchor_points = torch.cat(anchor_points).to(device)
    stride_tensor = torch.cat(stride_tensor).to(device)
    return anchors, anchor_points, num_anchors_list, stride_tensor
