"""Adaptive Training Sample Selection Assigner"""
from typing import List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.assigners.utils import dist_calculate, select_candidates_in_gts, select_highest_overlaps
from src.utils.iou import calculate_iou


class ATSSAssigner(nn.Module):
    def __init__(self, topk: int = 9, num_classes: int = 80) -> None:
        """instantiate assigner block

        :param topk: (int) hyper-parameter `topk` for assigner. default: 9
        :param num_classes: (int) number of object classes. default: 80 for COCO
        """
        super().__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_index = num_classes  # assign for background class usage

    def forward(
      self,
      anchor_bboxes: Tensor,
      n_level_bboxes: List[int],
      gt_bboxes: Tensor,
      gt_labels: Tensor,
      mask_gt: Tensor,
      pred_bboxes: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        """this code is based on `https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py`
        denote: B = self.bs, M = self.n_anchors, N = self.n_max_bboxes, C = self.num_classes

        :param anchor_bboxes: (Tensor[M, 4]) containing all anchor bboxes
        :param n_level_bboxes: (List[int]) specifying the number of anchor bboxes per FPN level
        :param gt_bboxes: (Tensor[B, N, 4]) containing ground truth bboxes for each image
        :param gt_labels: (Tensor[B, N, 1]) containing ground truth labels for each ground truth bbox
        :param mask_gt: (Tensor[B, N, 1]) indicating valid ground truth bboxes (1) and ignored ones (0)
        :param pred_bboxes: (Tensor[B, M, 4]) containing predicted bboxes from network (for soft label calculation)
        :return: a tuple contains of 3 tensor:
            - `target_labels` (Tensor[B, M]) Long tensor of assigned target labels (including background)
            - `target_bboxes` (Tensor[B, M, 4]) of assigned bboxes
            - `target_scores` (Tensor[B, M, C]) of one-hot encoded target class scores
            - `fg_mask` (Tensor[B, M]) boolean tensor indicating positive (True) and negative (False) anchor candidates
        """
        self.bs = gt_bboxes.size(0)
        self.n_anchors = anchor_bboxes.size(0)
        self.n_max_bboxes = gt_bboxes.size(1)

        # compute iou between all bboxes and gts
        overlaps = calculate_iou(gt_bboxes.reshape(-1, 4), anchor_bboxes)
        overlaps = overlaps.reshape(self.bs, -1, self.n_anchors)

        # compute center distances between all bboxes and gts
        distances, anchor_points = dist_calculate(gt_bboxes.reshape(-1, 4), anchor_bboxes)
        distances = distances.reshape(self.bs, -1, self.n_anchors)

        # select k candidates closest to the gt based on center distances
        is_in_candidate, candidate_indices = self.select_topk_candidates(distances, n_level_bboxes, mask_gt)

        # calculate threshold based on these candidates' IoU
        overlaps_threshold_per_gt, iou_candidates = self.threshold_calculator(
          is_in_candidate, candidate_indices, overlaps
        )

        # consider candidates whose IoU >= threshold as positive samples
        overlaps_threshold_per_gt = overlaps_threshold_per_gt.repeat(1, 1, self.n_anchors)
        is_positive = torch.where(
          iou_candidates > overlaps_threshold_per_gt, is_in_candidate, torch.zeros_like(is_in_candidate)
        )

        # check if candidates are within ground truth bboxes
        is_in_gts = select_candidates_in_gts(anchor_points, gt_bboxes)
        mask_positive = is_positive * is_in_gts * mask_gt

        # select highest IoU overlaps
        target_gt_index, fg_mask, mask_positive = select_highest_overlaps(mask_positive, overlaps, self.n_max_bboxes)

        # assigned target labels, bboxes, and scores
        target_labels, target_bboxes, target_scores = self.get_targets(gt_labels, gt_bboxes, target_gt_index, fg_mask)

        # soft label with IoU
        if pred_bboxes is not None:
            ious = calculate_iou(gt_bboxes, pred_bboxes) * mask_positive
            ious = ious.max(dim=-2)[0].unsqueeze(-1)
            target_scores = target_scores * ious

        return target_labels.long(), target_bboxes, target_scores, fg_mask.bool()

    def select_topk_candidates(
      self,
      distances: Tensor,
      num_level_bboxes: List[int],
      mask_gt: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """this function is used to select k nearest neighbor of each gt bboxes based on center distances.
        on each Pyramid level, for each gt, select k bboxes whose center are closest to the gt center,
        so we total select `k * l` anchor bboxes as candidates for each gt.
        denote: k = self.topk, l = pyramid level
            B = self.bs, N = self.n_max_bboxes, M = self.n_anchors

        :param distances: (Tensor[B, N, M]) containing distances between all ground truths and anchor bboxes
        :param num_level_bboxes: (List[int]) indicating number of anchor bboxes on each level
        :param mask_gt: (Tensor[B, N, 1]) mask tensor for selecting relevant ground truth bboxes
        :return: a tuple containing:
            - (Tensor[B, N, M]) indicates whether each candidate bbox belongs to the top-k closest ones for any considered ground truth bbox
            - (Tensor[B, N, k * l]) contains indices of selected candidate bboxes
        """
        # first transforms `mask_gt` to tensor whose dimensions are the same as `distances` tensor
        # by expanding the last dimension from 1 to 9 (aka. `self.topk`) and change dtype to `torch.bool`
        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()

        # split `distances` based on number of bboxes per level
        level_distances = torch.split(distances, num_level_bboxes, dim=-1)

        # a list to accumulate flags indicating if candidate bboxes belong to top-k closest for any considered ground truth bboxes
        is_in_candidate_list = []
        # a list used to keep track of candidate bboxes for each gt bbox
        candidate_indices = []

        start_index = 0
        for distances_per_level, bboxes_per_level in zip(level_distances, num_level_bboxes):
            # select top-k closest bboxes for each ground truth for current level
            selected_k = min(self.topk, bboxes_per_level)
            _, topk_indices_per_level = distances_per_level.topk(selected_k, dim=-1, largest=False)
            # update candidate indices with level offset
            candidate_indices.append(topk_indices_per_level + start_index)
            # filter indices based on `mask_gt` for considered ground truth bboxes
            topk_indices_per_level = torch.where(
              mask_gt, topk_indices_per_level, torch.zeros_like(topk_indices_per_level)
            )
            # create one-hot encoding to mark ground truth bbox membership
            is_in_candidate = F.one_hot(topk_indices_per_level, bboxes_per_level).sum(-2)
            # remove bboxes belonging to multiple ground truth bboxes (prevent duplicates)
            is_in_candidate = torch.where(is_in_candidate > 1, torch.zeros_like(is_in_candidate), is_in_candidate)
            # convert `is_in_candidate` to match `distances` dtype and accumulate per level
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            # update starting index for next level
            start_index = start_index + bboxes_per_level

        # concatenate results from all levels
        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_indices = torch.cat(candidate_indices, dim=-1)

        return is_in_candidate_list, candidate_indices

    def threshold_calculator(
      self,
      is_in_candidate: Tensor,
      candidate_indices: Tensor,
      overlaps: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """calculate `dynamic overlap threshold` for each ground truth bboxes
        based on its candidate bbox overlaps.
        denote: B = self.bs, N = self.n_max_bboxes, M = self.n_anchors

        :param is_in_candidate: (Tensor[B, N, M]) indicating which anchor bboxes are candidates for each ground truthb boxes
        :param candidate_indices: (Tensor[B, N, k * l]) indices of the candidate bboxes
        :param overlaps: (Tensor[B, N, M]) overlap scores (e.g. IoU) between all ground truth bboxes and anchor bboxes
        :return: a tuple of 2 tensor:
            - (Tensor[B, N, 1]) containing dynamic overlap thresholds for each ground truth bboxes
            - (Tensor[B, N, M]) containing filtered overlaps for candidate bboxes
        """
        # calculate maximum number of bboxes per batch
        num_gts = self.bs * self.n_max_bboxes

        # strictly filter overlaps to consider only valid candidates (e.g. top-k closest neighbors)
        _candidate_overlaps = torch.where(is_in_candidate > 0, overlaps, torch.zeros_like(overlaps))
        # reshape candidate indices for efficient indexing
        candidate_indices = candidate_indices.reshape(num_gts, -1)

        # create helper indices based on batch size and number of anchors
        # `assist_indices` create evenly spaced offsets, increasing by `self.n_anchors` for each ground truth bbox
        assist_indices = self.n_anchors * torch.arange(num_gts, device=candidate_indices.device)
        assist_indices = assist_indices[:, None]  # add new dimension in order to make it a column vector

        # combine indices for easier overlap lookup by adding the corresponding values from `assist_indices` to each row of `candidate_indices`
        # adding the offsets ensures that anchor indices for different ground truth bboxes don't overlap, even if they have the same values
        flatten_indices = candidate_indices + assist_indices
        # extract overlaps for valid candidate bboxes
        candidate_overlaps = _candidate_overlaps.reshape(-1)[flatten_indices]
        # reshape extracted overlaps to match ground truth structure
        candidate_overlaps = candidate_overlaps.reshape(self.bs, self.n_max_bboxes, -1)

        # calculate mean overlap per ground truth bbox
        overlaps_mean_per_gt = candidate_overlaps.mean(dim=-1, keepdim=True)
        # calculate standard deviation of overlaps per ground truth bbox
        overlaps_std_per_gt = candidate_overlaps.std(dim=-1, keepdim=True)
        # combine mean and standard deviation for dynamic thresholds
        overlaps_threshold_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_threshold_per_gt, _candidate_overlaps

    def get_targets(
      self,
      gt_labels: Tensor,
      gt_bboxes: Tensor,
      target_gt_index: Tensor,
      fg_mask: Tensor,
    ) -> Tuple[Tensor, Tensor, Tensor]:
        """constructs target labels, bboxes and scores for object detection training.
        denote: B = self.bs, N = self.n_max_bboxes, M = self.n_anchors, C = self.num_classes
        notes: bg_index is used for background class

        :param gt_labels: (Tensor[B, N, 1]) ground truth class labels
        :param gt_bboxes: (Tensor[B, N, 4]) ground truth bboxes
        :param target_gt_index: (Tensor[B, M]) indicating the assigned ground truth bbox index for each anchor
        :param fg_mask: (Tensor[B, M]) indicating positive candidate assignments
        :return: a tuple of 3 tensor:
            - `target_labels` (Tensor[B, M]) of assigned target labels (including background)
            - `target_bboxes` (Tensor[B, M, 4]) of assigned target bboxes
            - `target_scores` (Tensor[B, M, C]) of one-hot encoded target class scores
        """
        # calculate batch indices for proper indexing
        batch_index = torch.arange(self.bs, dtype=gt_labels.dtype, device=gt_labels.device)
        batch_index = batch_index[..., None]
        # use this to gather corresponding labels from `gt_labels`
        target_gt_index = (target_gt_index + batch_index * self.n_max_bboxes).long()

        # extract assigned target labels
        target_labels = gt_labels.flatten()[target_gt_index.flatten()]  # flatten for efficient indexing
        target_labels = target_labels.reshape(self.bs, self.n_anchors)  # reshape back to original format

        # assign background label (self.bg_index) to negative anchors (fg_mask == 0)
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.bg_index))

        # extract assigned target bboxes, similar process as for labels: flatten, gather and reshape
        target_bboxes = gt_bboxes.reshape(-1, 4)[target_gt_index.flatten()]
        target_bboxes = target_bboxes.reshape(self.bs, self.n_anchors, 4)

        # convert labels to one-hot encoded target scores
        # - add 1 for background class due to one-hot encoding requirements
        # - convert to long, create one-hot encoded target scores
        target_scores = F.one_hot(target_labels.long(), self.num_classes + 1).float()
        # remove the extra background class dimension at the end
        target_scores = target_scores[:, :, :self.num_classes]

        return target_labels, target_bboxes, target_scores
