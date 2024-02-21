"""Task-Aligned Learning Assigner"""
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.iou import calculate_iou

from .utils import select_candidates_in_gts, select_highest_overlaps


class TaskAlignedAssigner(nn.Module):
    def __init__(
      self, topk: int = 13, num_classes: int = 80, alpha: float = 1.0, beta: float = 6.0, eps: float = 1e-9
    ) -> None:
        """instantiate TAL Assigner building block

        :param topk: (int) the number of top matching candidates to consider for each ground-truth box. default: 13
        :param num_classes: (int) the number of object categories. default: 80 for COCO
        :param alpha: (float) weighted factor for classification score in the aligned metric. default: 1.0
        :param beta: (float) weighted factor for IoU value in the aligned metric. default: 6.0
        :param eps: (float) a small epsilon value for numerical stability. default: 1e-9
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_index = num_classes  # background class index
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

    @torch.no_grad()
    def forward(
      self,
      pred_scores: Tensor,
      pred_bboxes: Tensor,
      anchor_points: Tensor,
      gt_bboxes: Tensor,
      gt_labels: Tensor,
      mask_gt: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """assigns predicted boxes to ground-truth boxes.
        denote: B = self.bs, M = self.n_anchors, N = self.n_max_bboxes
            C = self.num_classes, K = self.topk
        this code is based on
            `https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py`

        :param pred_scores: (Tensor[B, M, C]) predicted class probabilities
        :param pred_bboxes: (Tensor[B, M, 4]) predicted bounding boxes
        :param anchor_points: (Tensor[M, 2]) anchor points
        :param gt_bboxes: (Tensor[B, N, 4]) ground truth bounding boxes
        :param gt_labels: (Tensor[B, N, 1]) ground truth labels
        :param mask_gt: (Tensor[B, N, 1]) indicating valid ground truth bboxes (1) and ignored ones (0)
        :return: a tuple contains 4 tensors:
            - `target_labels` (Tensor[B, M]) of assigned target labels
            - `target_bboxes` (Tensor[B, M, 4]) of assigned bounding boxes
            - `target_scores` (Tensor[B, M, C]) of one-hot encoded target class scores
            - `fg_mask` (Tensor[B, M]) boolean tensor indicating positive (True) and negative (False) anchor candidates
        """
        # get basic information about batch size and maximum number of anchors and ground-truth boxes per image
        self.bs, self.n_max_bboxes, _ = gt_bboxes.size()
        self.n_anchors = anchor_points.size(0)

        # early return with empty tensors if no ground-truth boxes are available
        if self.n_max_bboxes == 0:
            device = gt_bboxes.device
            return (
              torch.full_like(pred_scores[..., 0], self.bg_index, device=device),
              torch.zeros_like(pred_bboxes, device=device),
              torch.zeros_like(pred_scores, device=device),
              torch.zeros_like(pred_scores[..., 0], device=device),
            )

        # adjust loop strategy based on number of ground-truth boxes per image
        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_bboxes <= 100 else (self.bs, 1, 1)
        # initialize empty lists to store results for each mini-batch
        target_labels_list, target_bboxes_list, target_scores_list, fg_mask_list = [], [], [], []

        # loop over mini-batches for efficient processing
        for i in range(cycle):
            # select current mini-batch data
            start, end = i * step, (i+1) * step
            pred_scores_ = pred_scores[start:end, ...]
            pred_bboxes_ = pred_bboxes[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            mask_gt_ = mask_gt[start:end, ...]

            # calculate positive mask, align metric and overlaps for current mini-batch data
            mask_positive, align_metric, overlaps = self.get_positive_mask(
              pred_scores_, pred_bboxes_, gt_bboxes_, gt_labels_, anchor_points, mask_gt_
            )

            # select targets with highest overlaps and adjust positive mask
            target_gt_index, fg_mask, mask_positive = select_highest_overlaps(
              mask_positive, overlaps, self.n_max_bboxes
            )

            # assign target labels, bounding boxes, and scores based on selected targets
            target_labels, target_bboxes, target_scores = self.get_targets(
              gt_labels_, gt_bboxes_, target_gt_index, fg_mask
            )

            # normalize target scores based on align metric and overlaps
            align_metric *= mask_positive  # mask out align metric for non-positive anchors
            # calculate maximum align metric per anchor
            positive_align_metrics = align_metric.max(dim=-1, keepdim=True)[0]
            # calculate maximum overlap per anchor
            positive_overlaps = (overlaps * mask_positive).max(dim=-1, keepdim=True)[0]
            # calculate normalized align metric with help of eps value to avoid division by zero
            normalized_align_metric = (align_metric * positive_overlaps / (positive_align_metrics + self.eps))
            normalized_align_metric = normalized_align_metric.max(dim=-2)[0].unsqueeze(-1)
            target_scores = target_scores * normalized_align_metric  # normalize target scores using the normalized align metric

            # append current results to lists
            target_labels_list.append(target_labels)
            target_bboxes_list.append(target_bboxes)
            target_scores_list.append(target_scores)
            fg_mask_list.append(fg_mask)

        # concatenate results from all mini-batches
        target_labels = torch.cat(target_labels_list, dim=0)
        target_bboxes = torch.cat(target_bboxes_list, dim=0)
        target_scores = torch.cat(target_scores_list, dim=0)
        fg_mask = torch.cat(fg_mask_list, dim=0)

        return target_labels, target_bboxes, target_scores, fg_mask.bool()

    def get_positive_mask(
      self,
      pred_scores: Tensor,
      pred_bboxes: Tensor,
      gt_bboxes: Tensor,
      gt_labels: Tensor,
      anchor_points: Tensor,
      mask_gt: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """calculate the mask for positive samples (those assigned to ground-truth bounding boxes)
        denote: B = self.bs, M = self.n_anchors, N = self.n_max_bboxes
            C = self.num_classes, K = self.topk

        :param pred_scores: (Tensor[B, M, C]) predicted class probabilities for each anchor
        :param pred_bboxes: (Tensor[B, M, 4]) predicted bounding boxes for each anchor
        :param gt_bboxes: (Tensor[B, N, 4]) ground truth bounding boxes
        :param gt_labels: (Tensor[B, N, 1]) ground truth class labels
        :param anchor_points: (Tensor[M, 2]) anchors' center points
        :param mask_gt: (Tensor[B, N, 1]) mask for valid ground-truth bounding boxes
        :return: a tuple contains 3 tensors:
            - `mask_positive` (Tensor[B, N, M]) mask indicating positive samples
            - `align_metric` (Tensor[B, N, M]) Align metric for all anchors and ground-truth boxes
            - `overlaps` (Tensor[B, N, M]) IoU between predicted and ground-truth boxes
        """
        # calculate anchors' align metrics and overlaps
        align_metric, overlaps = self.get_bbox_metrics(pred_scores, pred_bboxes, gt_bboxes, gt_labels)
        # filter anchors within ground-truth boxes based on their locations
        in_gts_mask = select_candidates_in_gts(anchor_points, gt_bboxes)
        # select top-k candidates for each ground-truth box within high align metric
        mask_topk = self.select_topk_candidates(
          align_metric * in_gts_mask, topk_mask=mask_gt.repeat(1, 1, self.topk).bool()
        )
        # combine masks to identify the final positive anchors
        mask_positive = mask_topk * in_gts_mask * mask_gt

        return mask_positive, align_metric, overlaps

    def get_bbox_metrics(
      self,
      pred_scores: Tensor,
      pred_bboxes: Tensor,
      gt_bboxes: Tensor,
      gt_labels: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """calculates the align metric and IoU between predicted and ground-truth boxes
        denote: B = self.bs, M = self.n_anchors, N = self.n_max_bboxes
            C = self.num_classes, K = self.topk

        :param pred_scores: (Tensor[B, M, C]) predicted class probabilities
        :param pred_bboxes: (Tensor[B, M, 4]) predicted bounding boxes
        :param gt_bboxes: (Tensor[B, N, 4]) ground truth bounding boxes
        :param gt_labels: (Tensor[B, N, 1]) ground truth labels
        :return: a tuple contains 2 tensors:
            - `align_metric` (Tensor[B, N, M]) Align metric for all anchors and ground-truth boxes
            - `overlaps` (Tensor[B, N, M]) IoU between predicted and ground-truth boxes
        """
        # convert ground-truth labels to dtype
        gt_labels = gt_labels.to(torch.long)
        # get batch indices for each ground-truth box (for indexing predictions)
        batch_indices = torch.arange(self.bs).view(-1, 1).repeat(1, self.n_max_bboxes)
        # get classification scores for each ground-truth box by indexing predicted scores with batch and labels' indices
        bbox_cls_scores = pred_scores.permute(0, 2, 1)[batch_indices, gt_labels.squeeze(-1)]

        # get IoU scores between all ground truth and predicted bounding boxes
        overlaps = calculate_iou(gt_bboxes, pred_bboxes)
        # calculate the align metric based on class scores and IoU
        align_metric = bbox_cls_scores.pow(self.alpha) * overlaps.pow(self.beta)

        return align_metric, overlaps

    def select_topk_candidates(
      self, metrics: Tensor, topk_mask: Optional[Tensor] = None, largest: bool = True
    ) -> Tensor:
        """selects top-k candidates for each ground-truth box based on the given metric.
        denote: B = self.bs, M = self.n_anchors, N = self.n_max_bboxes
            C = self.num_classes, K = self.topk

        :param metrics: (Tensor[B, N, M]) containing the metrics for each ground-truth box
        :param topk_mask: (Tensor[B, N, K]) mask to filter valid ground-truth boxes before top-k selection
        :param largest: (bool, Optional) whether to select the largest or smallest k candidates. default: True
        :return: (Tensor[B, N, M]) mask indicating top-k selected candidates
        """
        # get top-k candidates based on specified metrics
        topk_metrics, topk_indices = metrics.topk(self.topk, dim=-1, largest=largest)
        if topk_mask is None:
            # if no mask provided, create one to include only candidates with metrics > eps
            topk_mask = (topk_metrics.max(dim=-1, keepdim=True)[0] > torch.tensor(self.eps)).tile([1, 1, self.topk])

        # replace invalid candidates' indices with 0 based on the mask
        topk_indices = torch.where(topk_mask, topk_indices, torch.zeros_like(topk_indices))
        # count occurrences of each index (across rows) to identify top-k candidates for each ground-truth box
        is_in_topk = F.one_hot(topk_indices, self.n_anchors).sum(dim=-2)
        # ensure each anchor is assigned to at most 1 ground-truth box
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
        # convert mask to the same dtype as the metrics
        return is_in_topk.to(metrics.dtype)

    def get_targets(
      self,
      gt_labels: Tensor,
      gt_bboxes: Tensor,
      target_gt_index: Tensor,
      fg_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """generates target labels, bounding boxes, and scores for assigned candidates
        denote: B = self.bs, M = self.n_anchors, N = self.n_max_bboxes
            C = self.num_classes, K = self.topk

        :param gt_labels: (Tensor[B, N, 1]) ground-truth class labels
        :param gt_bboxes: (Tensor[B, N, 4]) ground-truth bounding boxes
        :param target_gt_index: (Tensor[B, M]) indices of assigned ground-truth boxes for each anchor
        :param fg_mask: (Tensor[B, M]) mask indicating positive samples
        :return: a tuple contains 3 tensors:
            - `target_labels` (Tensor[B, M]) assigned class labels for each anchor
            - `target_bboxes` (Tensor[B, M, 4]) assigned ground-truth bounding boxes for each anchor
            - `target_scores` (Tensor[B, M, C]) target scores for each anchor
        """
        # get batch indices for each anchor
        batch_index = torch.arange(self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        # calculate absolute index for each assigned ground-truth box
        target_gt_index = target_gt_index + batch_index * self.n_max_bboxes
        # assigned target labels and clip negative labels to 0
        target_labels = gt_labels.long().flatten()[target_gt_index]
        target_labels[target_labels < 0] = 0

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape(-1, 4)[target_gt_index]

        # create one-hot encoded target scores
        target_scores = F.one_hot(target_labels, self.num_classes)
        # make foreground mask match dimension of scores and set background anchors' scores to 0
        fg_scores_mask = fg_mask[..., None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.full_like(target_scores, 0))

        return target_labels, target_bboxes, target_scores
