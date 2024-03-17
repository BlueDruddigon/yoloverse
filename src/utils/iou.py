from typing import Optional

import torch
from torch import Tensor


def _autocast(tensor: Tensor, scale: float = 1., dtype: Optional[str] = None) -> Tensor:
    """autocast float16 dtype

    :param tensor: tensor to be casted
    :param scale: scaling factor to prevent overflow. default: 1.
    :param dtype: dtype to cast (only 'fp16' affected). default: None
    :return: casted tensor
    """
    return (tensor / scale).half() if dtype == 'fp16' else tensor


def fp16_clamp(tensor: Tensor, min: Optional[float] = None, max: Optional[float] = None) -> Tensor:
    """utility to make clamp function supports fp16

    :param tensor: tensor to be clamped
    :param min: min value to clamp the tensor to. default: None
    :param max: max value to clamp the tensor to. default: None
    :return: clamped tensor
    """
    if not tensor.is_cuda and tensor.dtype == torch.float16:
        return tensor.float().clamp(min=min, max=max).half()
    return tensor.clamp(min=min, max=max)


def calculate_iou(
  boxes1: Tensor,
  boxes2: Tensor,
  mode: str = 'iou',
  is_aligned: bool = False,
  scale: float = 1.,
  dtype: Optional[str] = None
) -> Tensor:
    """calculate IoU between 2D bounding boxes.

    :param boxes1: (Tensor[B, M, 4]) bounding boxes in <x1, y1, x2, y2> format or
        (Tensor[B, M, 5]) bounding boxes in <x1, y1, x2, y2, score> format.
    :param boxes2: (Tensor[B, N, 4]) bounding boxes in <x1, y1, x2, y2> format or
        (Tensor[B, N, 5]) bounding boxes in <x1, y1, x2, y2, score> format.
    :param mode: IoU calculation mode (one of ['iou', 'iof', 'giou']). default: 'iou'
    :param is_aligned: if True, then M and N must be equal. default: False
    :param scale: scaling factor for autocast. default: 1.
    :param dtype: desired dtype to cast. default: None
    :return: a iou values tensor with shape (M, N) if `is_aligned` is False else (M, )
    """
    assert boxes1.size(-1) in [0, 4, 5]
    assert boxes2.size(-1) in [0, 4, 5]
    # remove unused `score` attribute
    if boxes1.size(-1) == 5:
        boxes1 = boxes1[..., :4]
    if boxes2.size(-1) == 5:
        boxes2 = boxes2[..., :4]

    # autocast boxes to desired dtype
    boxes1 = _autocast(boxes1, scale=scale, dtype=dtype)
    boxes2 = _autocast(boxes2, scale=scale, dtype=dtype)

    # calculate overlaps
    overlaps = bbox_overlaps(boxes1, boxes2, mode=mode, is_aligned=is_aligned)
    # resume float32 if desired dtype is float16
    if not overlaps.is_cuda and overlaps.dtype == torch.float16:
        overlaps = overlaps.float()

    return overlaps


def bbox_overlaps(
  boxes1: Tensor, boxes2: Tensor, mode: str = 'iou', is_aligned: bool = False, eps: float | Tensor = 1e-6
) -> Tensor:
    """calculate overlaps between 2 set of bounding boxes.
    this is modified version of `mmdet.core.bbox.iou_calculators.iou2d_calculator.bbox_overlaps`,
    with additional 'giou' mode and batch_size supports for bounding boxes.

    :param boxes1: (Tensor[B, M, 4]) bounding boxes in <x1, y1, x2, y2> format or empty tensor.
    :param boxes2: (Tensor[B, N, 4]) bounding boxes in <x1, y1, x2, y2> format or empty tensor.
    :param mode: the mode to be used for calculation (one of values: 'iou', 'iof' or 'giou'). default: 'iou'
    :param is_aligned: whether calculate the overlaps between each bbox of boxes1 and boxes2 or \
        between each aligned pair os boxes1 and boxes2. default: False
    :param eps: a value added to the denominator for numerical stability. default: 1e-6
    :return: a overlap values tensor with shape (M, N) if `is_aligned` is False else shape (M, )
    """
    assert mode in ['iou', 'iof', 'giou']
    # either the boxes are empty or the length of boxes' last dimension is 4
    assert boxes1.size(-1) == 4 or boxes1.size(0) == 0
    assert boxes2.size(-1) == 4 or boxes2.size(0) == 0

    # batch dimension must be the same
    assert boxes1.shape[:-2] == boxes2.shape[:-2]
    batch_shape = boxes1.shape[:-2]

    rows = boxes1.size(-2)
    cols = boxes2.size(-2)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return boxes1.new(batch_shape + (rows, )) if is_aligned else boxes1.new(batch_shape + (rows, cols))

    area1 = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    area2 = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    if is_aligned:
        lt = torch.max(boxes1[..., :2], boxes2[..., :2])  # [B, 2]
        rb = torch.min(boxes1[..., 2:], boxes2[..., 2:])  # [B, 2]

        wh = fp16_clamp(rb - lt, min=0)
        overlaps = wh[..., 0] * wh[..., 1]

        if mode == 'iof':
            union = area1
        else:
            union = area1 + area2 - overlaps
        if mode == 'giou':
            enclosed_lt = torch.min(boxes1[..., :2], boxes2[..., :2])
            enclosed_rb = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    else:
        lt = torch.max(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
        rb = torch.min(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

        wh = fp16_clamp(rb - lt, min=0)
        overlaps = wh[..., 0] * wh[..., 1]

        if mode == 'iof':
            union = area1[..., None]
        else:
            union = area1[..., None] + area2[..., None, :] - overlaps
        if mode == 'giou':
            enclosed_lt = torch.min(boxes1[..., :, None, :2], boxes2[..., None, :, :2])
            enclosed_rb = torch.max(boxes1[..., :, None, 2:], boxes2[..., None, :, 2:])

    # calculate ious
    eps = union.new_tensor([eps])
    union = torch.max(union, eps)
    ious = overlaps / union
    if mode in ['iou', 'iof']:
        return ious

    # calculate gious
    assert enclosed_lt and enclosed_rb
    enclosed_wh = fp16_clamp(enclosed_rb - enclosed_lt, min=0)
    enclosed_area = enclosed_wh[..., 0] * enclosed_wh[..., 1]
    enclosed_area = torch.max(enclosed_area, eps)
    gious = ious - (enclosed_area-union) / enclosed_area
    return gious
