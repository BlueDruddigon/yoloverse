from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.utils.bbox import bbox2dist

from .loss_iou import IoULoss


class BboxLoss(nn.Module):
    def __init__(self, num_classes: int, reg_max: int, use_dfl: bool = False, iou_type: str = 'giou') -> None:
        """this class implements bounding box loss

        :param num_classes: (int) number of object classes
        :param reg_max: (int) maximum value for the distance-based focal loss (DFL)
        :param use_dfl: (bool) whether to use DFL along with IoU loss. default: False
        :param iou_type: (str) type of IoU loss to use. default: 'giou'
            possible value: ['iou', 'giou', 'ciou', 'diou', 'siou']
        """
        super().__init__()
        self.num_classes = num_classes
        # create an IoU loss object within specified bbox format and IoU type
        self.iou_loss = IoULoss(bbox_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(
      self,
      pred_dist: Tensor,
      pred_bboxes: Tensor,
      anchor_points: Tensor,
      target_bboxes: Tensor,
      target_scores: Tensor,
      target_scores_sum: Tensor,
      fg_mask: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        """calculates the bounding box loss for a batch of predictions and targets

        :param pred_dist: (Tensor[B, M, 4]) predicted distance distribution for each anchor points
        :param pred_bboxes: (Tensor[B, M, 4]) predicted bounding boxes
        :param anchor_points: (Tensor[M, 2]) anchor points used for prediction
        :param target_bboxes: (Tensor[B, M, 4]) ground-truth bounding boxes
        :param target_scores: (Tensor[B, M, C]) ground-truth objectness scores
        :param target_scores_sum: (float) sum of objectness scores for each image
        :param fg_mask: (Tensor[B, M]) foreground mask indicating positive samples
        :return: a tuple contains:
            - `loss_iou` (float) the IoU loss
            - `loss_dfl` (float) the DFL loss
        """
        # select positive samples' mask
        num_positive = fg_mask.sum()
        if num_positive > 0:
            # compute mask for positive samples
            bbox_mask = fg_mask.unsqueeze(-1).repeat(1, 1, 4)
            # select positive predictions and targets
            pred_bboxes_positive = torch.masked_select(pred_bboxes, bbox_mask).reshape(-1, 4)
            target_bboxes_positive = torch.masked_select(target_bboxes, bbox_mask).reshape(-1, 4)
            # calculate weights based on target objectness scores
            bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            # calculate IoU loss with weights
            loss_iou = self.iou_loss(pred_bboxes_positive, target_bboxes_positive) * bbox_weight

            # sum the loss over all samples
            loss_iou = loss_iou.sum()
            if target_scores_sum > 1:  # normalize the loss by total objectness scores if necessary
                loss_iou = loss_iou / target_scores_sum

            if self.use_dfl:  # if using DFL loss
                # compute mask for positive predictions
                dist_mask = fg_mask.unsqueeze(-1).repeat(1, 1, (self.reg_max + 1) * 4)
                # select positive distance predictions
                pred_dist_positive = torch.masked_select(pred_dist, dist_mask).reshape(-1, 4, self.reg_max + 1)
                # convert ground-truth boxes to distance representation
                target_ltrb = bbox2dist(anchor_points, target_bboxes, reg_max=self.reg_max)
                # select positive ground-truth distances
                target_ltrb_positive = torch.masked_select(target_ltrb, bbox_mask).reshape(-1, 4)
                # calculate DFL loss with weights
                loss_dfl = self._dfl_loss(pred_dist_positive, target_ltrb_positive) * bbox_weight

                # sum the loss across samples
                loss_dfl = loss_dfl.sum()
                if target_scores_sum > 1:  # normalize if needed
                    loss_dfl = loss_dfl / target_scores_sum
            else:  # set the loss to 0 if DFL wasn't used
                loss_dfl = pred_dist.sum() * 0.
        else:  # no positive samples
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _dfl_loss(self, pred_dist: Tensor, target: Tensor) -> Tensor:
        """calculates the distance-based focal loss (DFL) for a batch of predictions and targets
        denote: R = self.reg_max. the first dimension of `pred_dist` is flexible caused by
            number of positive samples' assigned by the label assigner

        :param pred_dist: (Tensor[-1, 4, R]) predicted distance distribution for each anchor points
        :param target: (Tensor[-1, 4]) ground-truth distances in <left, top, right, bottom> format
        :return: (float) DFL loss for each prediction
        """
        # convert target values to long for indexing and right values for bining
        target_left = target.to(torch.long)
        target_right = target_left + 1

        # calculate weights based on distance between predicted and target bins
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left

        # apply cross-entropy loss to each bin separately
        loss_left = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none')
        loss_right = F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none')

        # combine losses weighted by distance from target bin
        loss = loss_left.view(target_left.shape) * weight_left + loss_right.view(target_right.shape) * weight_right

        # return the mean loss across all dimensions except the last (samples)
        return loss.mean(dim=-1, keepdim=True)
