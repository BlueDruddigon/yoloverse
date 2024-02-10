from typing import List, Sequence, Tuple, Union

import torch


def generate_anchors(
  feats: Sequence[torch.Tensor],
  fpn_strides: Sequence[int],
  grid_cell_size: float = 5.0,
  grid_cell_offset: float = 0.5,
  device: Union[str, torch.device] = 'cpu',
  is_eval: bool = False
) -> Union[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, List[int], torch.Tensor]]:
    """Generate anchor points from fpn feature maps"""
    anchors = []
    anchor_points = []
    stride_tensor = []
    num_anchors_list = []
    assert feats is not None

    if is_eval:
        for i, stride in enumerate(fpn_strides):
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
