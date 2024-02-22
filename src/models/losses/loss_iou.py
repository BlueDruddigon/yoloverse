import math

import torch
import torch.nn as nn


class IoULoss(nn.Module):
    def __init__(
      self, bbox_format: str = 'xywh', iou_type: str = 'ciou', reduction: str = 'none', eps: float = 1e-7
    ) -> None:
        """this class implements the IoU loss, which is a metric used to measure the overlaps area between two bounding boxes

        :param bbox_format: (str) the format of the bounding boxes. default: 'xywh'
        :param iou_type: (str) indicating the IoU type to calculate. default: 'ciou'
            possible choice: ['iou', 'giou', 'diou', 'ciou', 'siou']
        :param reduction: (str) the reduction type to apply to the loss. default: 'none'
            possible choice: ['sum', 'none', 'mean']
        :param eps: (float) a small value to prevent division by zero. default: 1e-7
        """
        super().__init__()
        assert bbox_format in ['xyxy', 'xywh']
        assert reduction in ['mean', 'sum', 'none']
        self.bbox_format = bbox_format
        self.iou_type = iou_type.lower()  # convert to lowercase for consistency
        self.reduction = reduction
        self.eps = eps

    def forward(self, box1: torch.Tensor, box2: torch.Tensor) -> torch.Tensor:
        """calculates the IoU loss between 2 batches of bounding boxes

        :param box1: (Tensor[N, 4]) representing the first batch of bounding boxes.
            if `bbox_format` is 'xyxy', the tensor should have <x1, y1, x2, y2> format
            if `bbox_format` is 'xywh', the tensor should have <cx, cy, w, h> format
        :param box2: (Tensor[N, 4]) representing the second batch of bounding boxes.
            the format should be the same.
        :return: (Tensor[N, 1]) the IoU loss between 2 batches of bounding boxes (after applied reduction)
        """
        # check if the number of boxes in `box1` and `box2` is the same
        if box1.shape[0] != box2.shape[0]:
            box2 = box2.T  # transpose `box2` if not
            if self.bbox_format == 'xyxy':
                # split tensors into coordinates if `bbox_format` is 'xyxy'
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)
            elif self.bbox_format == 'xywh':
                # if the `bbox_format` is 'xywh', then convert it to 'xyxy'
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w/2, b1_x1 + b1_w/2
                b1_y1, b1_y2 = b1_y1 - b1_h/2, b1_y1 + b1_h/2
                b2_x1, b2_x2 = b2_x1 - b2_w/2, b2_x1 + b2_w/2
                b2_y1, b2_y2 = b2_y1 - b2_h/2, b2_y1 + b2_h/2
            else:
                raise ValueError
        else:
            # this process is same as the above, except transpose step
            if self.bbox_format == 'xyxy':
                # split tensors into coordinates if `bbox_format` is 'xyxy'
                b1_x1, b1_y1, b1_x2, b1_y2 = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_x2, b2_y2 = torch.split(box2, 1, dim=-1)
            elif self.bbox_format == 'xywh':
                # if the `bbox_format` is 'xywh', then convert it to 'xyxy'
                b1_x1, b1_y1, b1_w, b1_h = torch.split(box1, 1, dim=-1)
                b2_x1, b2_y1, b2_w, b2_h = torch.split(box2, 1, dim=-1)
                b1_x1, b1_x2 = b1_x1 - b1_w/2, b1_x1 + b1_w/2
                b1_y1, b1_y2 = b1_y1 - b1_h/2, b1_y1 + b1_h/2
                b2_x1, b2_x2 = b2_x1 - b2_w/2, b2_x1 + b2_w/2
                b2_y1, b2_y2 = b2_y1 - b2_h/2, b2_y1 + b2_h/2
            else:
                raise ValueError

        # calculate intersection area
        inter = (torch.min(b1_x2, b2_x2) - torch.max(b1_x1, b2_x1)).clamp(0) * \
            (torch.min(b1_y2, b2_y2) - torch.max(b1_y1, b2_y1)).clamp(0)

        # calculate union area and IoU score
        w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + self.eps
        w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + self.eps
        union = w1*h1 + w2*h2 - inter + self.eps
        iou = inter / union

        # calculate convex width and height for advanced `iou_type`
        cw = torch.max(b1_x2, b2_x2) - torch.min(b1_x1, b2_x1)  # convex width
        ch = torch.max(b1_y2, b2_y2) - torch.min(b1_y1, b2_y1)  # convex height

        # calculate IoU based on the chosen type
        if self.iou_type == 'giou':  # Generalized IoU (GIoU)
            c_area = cw*ch + self.eps  # convex area
            iou = iou - (c_area-union) / c_area
        elif self.iou_type in ['diou', 'ciou']:
            c2 = cw ** 2 + ch ** 2 + self.eps  # convex diagonal squared
            # center distance squared
            rho2 = ((b2_x1 + b2_x2 - b1_x1 - b1_x2) ** 2 + (b2_y1 + b2_y2 - b1_y1 - b1_y2) ** 2) / 4
            if self.iou_type == 'diou':  # distance-based IoU (DIoU)
                iou = iou - rho2/c2
            else:
                assert self.iou_type == 'ciou'  # complete IoU (CIoU)
                # the `v` term captures the `difference in aspect ratios` between the two bounding boxes
                v = (4 / math.pi ** 2) * torch.pow(torch.atan(w2 / h2) - torch.atan(w1 / h1), 2)
                with torch.no_grad():
                    # the weighted factor for the `v` term
                    alpha = v / (v - iou + (1 + self.eps))
                iou = iou - (rho2/c2 + v*alpha)
        elif self.iou_type == 'siou':  # Scale-Invariant IoU (SIoU)
            # SIoU loss https://arxiv.org/pdf/2205.12740.pdf
            s_cw = (b2_x1+b2_x2-b1_x1-b1_x2) * 0.5 + self.eps  # scaled convex width
            s_ch = (b2_y1+b2_y2-b1_y1-b1_y2) * 0.5 + self.eps  # scaled convex height
            # scaled distance metrics
            sigma = torch.pow(s_cw ** 2 + s_ch ** 2, 0.5)
            sin_alpha_1 = torch.abs(s_cw) / sigma
            sin_alpha_2 = torch.abs(s_ch) / sigma
            # determine threshold and dominant angle
            threshold = pow(2, 0.5) / 2
            sin_alpha = torch.where(sin_alpha_1 > threshold, sin_alpha_2, sin_alpha_1)
            # calculate angle cost
            angle_cost = torch.cos(torch.arcsin(sin_alpha) * 2 - math.pi / 2)
            # calculate distance cost
            rho_x = (s_cw / cw) ** 2  # normalized distance ratios (width)
            rho_y = (s_ch / ch) ** 2  # normalized distance ratios (height)
            gamma = angle_cost - 2  # hyper-parameter
            distance_cost = 2 - torch.exp(gamma * rho_x) - torch.exp(gamma * rho_y)
            # calculate shape cost
            omiga_w = torch.abs(w1 - w2) / torch.max(w1, w2)  # normalized difference for width
            omiga_h = torch.abs(h1 - h2) / torch.max(h1, h2)  # normalized difference for height
            shape_cost = torch.pow(1 - torch.exp(-1 * omiga_w), 4) + torch.pow(1 - torch.exp(-1 * omiga_h), 4)
            # update IoU score
            iou = iou - 0.5 * (distance_cost+shape_cost)

        # calculate the loss
        loss = 1. - iou
        # apply reduction
        if self.reduction == 'sum':
            loss = loss.sum()
        elif self.reduction == 'mean':
            loss = loss.mean()

        return loss


def pairwise_bbox_iou(box1: torch.Tensor, box2: torch.Tensor, bbox_format: str = 'xywh') -> torch.Tensor:
    """calculate pairwise iou of 2 bounding box.
    this code is based on `https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py`

    :param box1: (Tensor[M, 4]) first bounding box
    :param box2: (Tensor[N, 4]) second bounding box
    :param bbox_format: (str) the format of the bounding box
    :return: (Tensor[M, N]) of calculated pairwise IoU scores
    """
    if bbox_format == 'xyxy':
        lt = torch.max(box1[:, None, :2], box2[:, :2])
        rb = torch.min(box1[:, None, 2:], box2[:, 2:])
        area1 = torch.prod(box1[:, 2:] - box1[:, :2], dim=1)
        area2 = torch.prod(box2[:, 2:] - box2[:, :2], dim=1)
    elif bbox_format == 'xywh':
        lt = torch.max((box1[:, None, :2] - box1[:, None, 2:] / 2), (box2[:, :2] - box2[:, 2:] / 2))
        rb = torch.min((box1[:, None, :2] + box1[:, None, 2:] / 2), (box2[:, :2] + box2[:, 2:] / 2))
        area1 = torch.prod(box1[:, 2:], dim=1)
        area2 = torch.prod(box2[:, 2:], dim=1)
    else:
        raise ValueError

    valid = (lt < rb).to(lt.dtype).prod(dim=2)
    inter = torch.prod(rb - lt, dim=2) * valid
    return inter / (area1[:, None] + area2 - inter)
