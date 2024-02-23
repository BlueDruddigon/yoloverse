from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.assigners.anchor_generator import generate_anchors
from src.assigners.atss import ATSSAssigner
from src.assigners.tal import TaskAlignedAssigner
from src.utils.bbox import bbox2dist, dist2bbox, xywh2xyxy

from .loss_iou import IoULoss


class ComputeLoss(nn.Module):
    def __init__(
      self,
      fpn_strides: List[int] = [8, 16, 32],
      grid_cell_size: float = 5.0,
      grid_cell_offset: float = 0.5,
      num_classes: int = 80,
      origin_image_size: int = 640,
      warmup_epoch: int = 4,
      use_dfl: bool = True,
      reg_max: int = 16,
      iou_type: str = 'giou',
      loss_weight: Dict[str, float] = {
        'class': 1.0,
        'iou': 2.5,
        'dfl': 0.5
      }
    ) -> None:
        """this class implements the computation of loss for YOLOv6, which is

        :param fpn_strides: (List[int]) list of feature map strides for different FPN levels. default: [8, 16, 32]
        :param grid_cell_size: (float) size of a grid cell in the image. default: 5.0
        :param grid_cell_offset: (float) offset of the grid cell. default: 0.5
        :param num_classes: (int) number of object classes. default: 80
        :param origin_image_size: (int) original image resolution. default: 640
        :param warmup_epoch: number of warm-up epochs for using different assignment strategies. default: 4
        :param use_dfl: (bool) whether to use distance focal loss. default: True
        :param reg_max: (int) maximum value for distance prediction. default: 16
        :param iou_type: (str) type of IoU loss to use. default: 'giou'
            possible values: ['iou', 'giou', 'ciou', 'diou', 'siou']
        :param loss_weight: (Dict[str, float]) dictionary containing weights for different loss components.
            default: {'class': 1.0, 'iou': 2.5, 'dfl': 0.5}
        """
        super().__init__()
        self.fpn_strides = fpn_strides
        # cached feature map sizes (empty initially)
        self.cached_feat_sizes = [torch.Size([0, 0]) for _ in fpn_strides]
        self.cached_anchors = None  # cached anchors (not calculated yet)
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.origin_image_size = origin_image_size

        self.warmup_epoch = warmup_epoch
        # assigner for warm-up phase
        self.warmup_assigner = ATSSAssigner(topk=9, num_classes=self.num_classes)
        # assigner for regular training
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.iou_type = iou_type
        # learned parameter for distance projection
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.varifocal_loss = VariFocalLoss()
        self.bbox_loss = BboxLoss(self.num_classes, reg_max=self.reg_max, use_dfl=self.use_dfl, iou_type=self.iou_type)
        self.loss_weight = loss_weight

    def forward(
      self,
      outputs: Tuple[List[Tensor], Tensor, Tensor],
      targets: Tensor,
      epoch_num: int,
      step_num: int,
      batch_height: int,
      batch_width: int,
    ):
        """calculates the loss for a given batch
        denote: X = number of actual samples, C = number of object categories,
            B = batch size, M = maximum number of anchors,
            P = 4 * (self.reg_max + 1)  # the output of regression branch in network's head
            P = 68 when using DFL, otherwise P = 4,

        :param outputs: (Tuple) tuple containing network outputs (feature maps, predicted scores, predicted distances)
            - `feats`: (Tuple[Tensor]) representing list of feature maps within FPN levels
            - `pred_scores` (Tensor[B, M, C]) predicted scores
            - `pred_dist` (Tensor[B, M, P]) predicted distances distribution
        :param targets: (Tensor[X, 6]) ground-truth targets
        :param epoch_num: (int) current epoch number
        :param step_num: (int) current step number within epoch
        :param batch_height: (int) batch's image height
        :param batch_width: (int) batch's image width
        :return: a tuple contains:
            - `loss` (float) the total loss
            - (Tensor[3]) containing individual loss components: `IoU`, `DFL` and `class`
        """
        feats, pred_scores, pred_dist = outputs  # network's output unpack

        # check if feature map sizes have changed and re-calculate anchors if necessary
        if all(feat.shape[2:] == cfsize for feat, cfsize in zip(feats, self.cached_feat_sizes)):
            anchors, anchor_points, n_anchors_list, stride_tensor = self.cached_anchors
        else:
            self.cached_feat_sizes = [feat.shape[2:] for feat in feats]
            anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
              feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device
            )
            self.cached_anchors = anchors, anchor_points, n_anchors_list, stride_tensor

        assert pred_scores.type() == pred_dist.type()
        # calculate scale tensor based on batch dimensions
        gt_bboxes_scale = torch.tensor([batch_width, batch_height, batch_width, batch_height]).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # preprocess target ground-truth
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[..., :1]  # extract ground-truth labels
        gt_bboxes = targets[..., 1:]  # extract ground-truth bounding boxes (xyxy format)
        mask_gt = (gt_bboxes.sum(dim=-1, keepdim=True) > 0).float()  # create mask for valid ground-truth boxes

        # convert anchor points to relative coordinates
        anchor_points_s = anchor_points / stride_tensor
        # decode predictions to bounding boxes
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_dist)

        # choose appropriate assignment strategy based on current epoch
        try:
            if epoch_num < self.warmup_epoch:  # use warmup assigner during initial epochs
                target_labels, target_bboxes, target_scores, fg_mask = self.warmup_assigner(
                  anchors,
                  n_anchors_list,
                  gt_bboxes,
                  gt_labels,
                  mask_gt,
                  pred_bboxes.detach() * stride_tensor,
                )
            else:  # use formal assigner after warmup period
                target_labels, target_bboxes, target_scores, fg_mask = self.formal_assigner(
                  pred_scores.detach(),
                  pred_bboxes.detach() * stride_tensor,
                  anchor_points,
                  gt_bboxes,
                  gt_labels,
                  mask_gt,
                )
        except RuntimeError:  # handle Out-of-Memory (OOM) error and switch to CPU mode if necessary
            print(
              'OOM RuntimeError is raised due to the huge memory cost during label assignment. '
              'CPU mode is applied in this batch. If you want to avoid this issue, '
              'try to reduce the batch size or image size.'
            )
            torch.cuda.empty_cache()
            print('------------CPU Mode for This Batch-------------')
            if epoch_num < self.warmup_epoch:  # warmup assignment
                # move tensors to CPU due to OOM error
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_bboxes = gt_bboxes.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                # perform assignment on CPU
                target_labels, target_bboxes, target_scores, fg_mask = self.warmup_assigner(
                  _anchors, _n_anchors_list, _gt_bboxes, _gt_labels, _mask_gt, _pred_bboxes * _stride_tensor
                )
            else:  # formal assignment
                # move tensors to CPU due to OOM error
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _mask_gt = mask_gt.cpu().float()

                # perform assignment on CPU
                target_labels, target_bboxes, target_scores, fg_mask = self.formal_assigner(
                  _pred_scores, _pred_bboxes * _stride_tensor, _anchor_points, _gt_bboxes, _gt_labels, _mask_gt
                )

            # move results back to current using device for further processing
            target_labels = target_labels.to(device=pred_scores.device)
            target_bboxes = target_bboxes.to(device=pred_scores.device)
            target_scores = target_scores.to(device=pred_scores.device)
            fg_mask = fg_mask.to(device=pred_scores.device)

        # periodically release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale target bounding boxes based on stride
        target_bboxes = target_bboxes / stride_tensor

        # replace ground-truth labels with background class for ignored boxes
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]  # including background class
        # calculate classification loss
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid division by zero error, as it will cause loss value to be `inf` or `nan`
        # if `target_scores_sum` is 0, `loss_cls` equals to 0 also
        if target_scores_sum > 1:
            loss_cls = loss_cls / target_scores_sum

        # calculate bounding box loss (IoU and distance-based focal loss)
        loss_iou, loss_dfl = self.bbox_loss(
          pred_dist, pred_bboxes, anchor_points_s, target_bboxes, target_scores, target_scores_sum, fg_mask
        )

        # calculate total loss with weighted combination of `loss_cls`, `loss_iou` and `loss_dfl`
        loss = self.loss_weight['class'] * loss_cls + \
            self.loss_weight['iou'] * loss_iou + \
            self.loss_weight['dfl'] * loss_dfl

        # return total loss and individual loss components
        return loss, torch.cat([
          (self.loss_weight['iou'] * loss_iou).unsqueeze(0),
          (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
          (self.loss_weight['class'] * loss_cls).unsqueeze(0),
        ]).detach()

    def preprocess(self, targets: Tensor, batch_size: int, scale_tensor: Tensor) -> Tensor:
        """pre-process the ground-truth targets
        denote: B = batch size, S = maximum number of target objects per batch, X = actual number of targets
            6 is represented for <batch_index, cls, cx, cy, w, h>
            4 is represented for <h, w, h, w>
            5 is represented for <cls, x1, y1, x2, y2>

        :param targets: (Tensor[X, 6]) the ground-truth targets tensor
        :param batch_size: (int) batch size
        :param scale_tensor: (Tensor[4]) tensor to scale target coordinates
        :return: (Tensor[B, S, 5]) preprocessed targets tensor
        """
        targets_list = np.zeros((batch_size, 1, 5)).tolist()  # empty list to store
        for item in targets.cpu().numpy().tolist():
            # the first dimension is represented for batch size, and 5 values after are <cls, cx, cy, w, h> format
            targets_list[int(item[0])].append(item[1:])
        max_len = max(len(t) for t in targets_list)  # max length of `targets_list`'s sublist
        # pad all sublist with [[-1, 0, 0, 0, 0]] to ensure that every sublist has same length. [-1, 0, 0, 0, 0] is the dummy object
        targets = torch.from_numpy(
          np.array(list(map(lambda x: x + [[-1, 0, 0, 0, 0]] * (max_len - len(x)), targets_list)))[:, 1:, :]
        ).to(targets.device)
        # re-scale coordinates from [0, 1] for both height and width to [0, H] for height and [0, W] for width. denote: H and W is image's original resolution
        batch_target = targets[..., 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)  # convert to 'xyxy' format for bounding boxes
        return targets

    def bbox_decode(self, anchor_points: Tensor, pred_dist: Tensor) -> Tensor:
        """decode predicted bounding boxes from anchor points and distances
        denote: M = maximum number of anchor boxes, B = batch size
            P = 4 * (self.reg_max + 1)  # the output of regression branch in network's head
            P = 68 when using DFL, otherwise P = 4

        :param anchor_points: (Tensor[M, 2]) anchor points tensor
        :param pred_dist: (Tensor[B, M, P]) predicted distances tensor
        :return: (Tensor[B, M, 4]) decoded bounding boxes tensor
        """
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            # apply softmax and project distances using pre-defined values
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1)
            pred_dist = pred_dist.matmul(self.proj.to(pred_dist.device))

        # calculate bounding box coordinates based on distances and anchors center points
        return dist2bbox(pred_dist, anchor_points)


class VariFocalLoss(nn.Module):
    def __init__(self) -> None:
        """ VariFocal Loss initialization """
        super().__init__()

    def forward(
      self, pred_score: Tensor, gt_score: Tensor, label: Tensor, alpha: float = 0.75, gamma: float = 2.0
    ) -> Tensor:
        """calculate the varifocal loss.
        denote: B = batch size, M = maximum number of anchor boxes, C = number of classes

        :param pred_score: (Tensor[B, M, C]) the predicted class probabilities
        :param gt_score: (Tensor[B, M, C]) the score of assigned targets
        :param label: (Tensor[B, M, C]) the ground-truth class label
        :param alpha: (float) factor controlling the weight of easy samples. default: 0.75
        :param gamma: (float) factor controlling the focusing ability. default: 2.0
        :return: (float) the calculated VariFocal loss
        """
        weight = alpha * pred_score.pow(gamma) * (1-label) + gt_score*label
        with torch.autocast(pred_score.device.type, enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


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
            number of positive samples' assigned by the label assigner.
            so as to first dimension of `targets` tensor

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
