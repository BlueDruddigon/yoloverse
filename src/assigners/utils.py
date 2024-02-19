"""Utilities for label assignment modules"""
from typing import Tuple

import torch
import torch.nn.functional as F


def dist_calculate(gt_boxes: torch.Tensor, anchor_boxes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """compute L2 distances from center points between all bboxes and gts by L2
    denote: M = number of anchor bboxes, N = number of max bboxes per image

    :param gt_boxes: (Tensor[N, 4]) ground-truth bboxes, where `N = batch_size * n_max_boxes`
    :param anchor_boxes: (Tensor[M, 4]) anchor bboxes, where `M = n_total_anchors`
    :return: a tensor contains pairwise L2 distances and a tensor that stores anchor points
    """
    # get ground-truth boxes' center
    gt_centers = (gt_boxes[..., :2] + gt_boxes[..., 2:]) / 2.

    # get anchor boxes' center
    anchor_centers = (anchor_boxes[..., 2:] + anchor_boxes[..., 2:]) / 2.

    # calculate pairwise L2 distances
    distances = (gt_centers.unsqueeze(1) - anchor_centers.unsqueeze(0)).pow(2).sum(-1).sqrt()

    return distances, anchor_centers


def select_candidates_in_gts(xy_centers: torch.Tensor, gt_bboxes: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """select the positive anchor points whose centers in assigned ground truth bboxes
    denote: B = batch size, M = number of anchor bboxes, N = number of max bboxes per image

    :param xy_centers: (Tensor[M, 2]) anchor point center coordinates (xy format)
    :param gt_bboxes: (Tensor[B, N, 4]) ground truth bboxes (xyxy format)
    :param eps: epsilon value for numerical stability to avoid division by zero. default: 1e-9
    :return: (Tensor[B, N, M]) boolean tensor indicating whether a candidate bboxes is within assigned ground truth bboxes
    """
    # extract basic dimension
    n_anchors = xy_centers.size(0)
    bs, n_max_bboxes, _ = gt_bboxes.size()

    # reshape ground truth bboxes for easier manipulation
    gt_bboxes = gt_bboxes.reshape(-1, 4)

    # repeat the centers of candidate bboxes to match the shape of ground truth bboxes
    xy_centers = xy_centers[None, ...].repeat(bs * n_max_bboxes, 1, 1)
    # repeat the corners of ground truth bboxes to match the shape of candidate bboxes
    gt_bboxes_lt = gt_bboxes[..., None, :2].repeat(1, n_anchors, 1)
    gt_bboxes_rb = gt_bboxes[..., None, 2:].repeat(1, n_anchors, 1)

    # compute the distances from candidate bboxes to ground truth bboxes
    bbox_lt = xy_centers - gt_bboxes_lt
    bbox_rb = gt_bboxes_rb - xy_centers

    # concatenate the computed distances to get bbox deltas
    bbox_deltas = torch.cat([bbox_lt, bbox_rb], dim=-1)

    # reshape bounding box deltas to the original shape
    bbox_deltas = bbox_deltas.reshape(bs, n_max_bboxes, n_anchors, -1)
    # check if the minimum distance between candidate bboxes and ground truth bboxes is greater than eps
    return (bbox_deltas.min(dim=-1)[0] > eps).to(gt_bboxes.dtype)


def select_highest_overlaps(
  mask_positive: torch.Tensor,
  overlaps: torch.Tensor,
  n_max_bboxes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """if an anchor bbox is assigned to multiple gts,
    the one with the highest IoU will be selected.
    denote: B = batch size, N = number of max bboxes per image, M = number of anchor bboxes

    :param mask_positive: (Tensor[B, N, M]) boolean tensor indicating positive candidate assignments.
        True at (i, j, k) indicates that anchor `j` in batch `i` is a candidate for gt bbox `k`.
    :param overlaps: (Tensor[B, N, M]) containing overlap scores between anchors and ground truth bboxes
    :param n_max_bboxes: (int) maximum number of ground truth bboxes per image
    :return: a tuple contains 3 tensor:
        - `target_gt_index` (Tensor[B, M]) index of the highest overlapping ground truth bbox for each anchor
        - `fg_mask` (Tensor[B, M]) boolean tensor to track positive candidate assignments
        - `mask_positive` (Tensor[B, N, M]) the updated mask with highest overlaps
    """
    fg_mask = mask_positive.sum(dim=-2)  # count positive assigments per anchor across gt bboxes
    if fg_mask.max() > 1:  # check if any anchor has more than one positive assignment
        # identify anchors with multiple candidates
        mask_multiple_gts = (fg_mask.unsqueeze(1) > 1).repeat(1, n_max_bboxes, 1)
        # find index of highest overlap for each anchor
        max_overlaps_index = overlaps.argmax(dim=1)
        # create one-hot encoding of highest overlap positions
        is_max_overlaps = F.one_hot(max_overlaps_index, n_max_bboxes)
        # permute and cast to match dtype
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        # update mask with only highest overlaps
        mask_positive = torch.where(mask_multiple_gts, is_max_overlaps, torch.zeros_like(is_max_overlaps))
        # re-calculate positive assignment after update mask
        fg_mask = mask_positive.sum(dim=-2)

    # find final target gt index and return results
    target_gt_index = mask_positive.argmax(dim=-2)
    return target_gt_index, fg_mask, mask_positive
