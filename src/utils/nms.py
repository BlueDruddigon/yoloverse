import os
import time
from typing import List, Optional

import cv2
import numpy as np
import torch
import torchvision

from .bbox import xywh2xyxy
from .logger import LOGGER

torch.set_printoptions(linewidth=320, precision=5, profile='long')
np.set_printoptions(linewidth=320, formatter={'float_kind': '{:11.5g}'.format})  # format short g, %precision=5
cv2.setNumThreads(0)  # prevent OpenCV from multithreading (incompatible with PyTorch DataLoader)
CPUS = os.cpu_count()  # num cpus
assert CPUS is not None
os.environ['NUMEXPR_MAX_THREADS'] = str(min(CPUS, 8))  # NumExpr max threads


def non_max_suppression(
  prediction,
  conf_thres: float = 0.25,
  iou_thres: float = 0.45,
  classes: Optional[List[int]] = None,
  agnostic: bool = False,
  multi_label: bool = False,
  max_det: int = 300
) -> List[torch.Tensor]:
    """performs non-maximum suppression (NMS) on inference results.
    This code is borrowed from: `https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775

    :param prediction: (Tensor) the predicted bounding boxes, `shape=[N, 5 + num_classes]`, where `N` is the number of bounding boxes.
    :param conf_thres: (float) confidence threshold. default: 0.25
    :param iou_thres: (float) IoU (Intersection over Union) threshold. default: 0.45
    :param classes: (list | None) list of classes to filter the boxes, only boxes with categories in this list will be kept. default: None
    :param agnostic: (bool) whether to do class-independent NMS. default: False
    :param multi_label: (bool) whether to allow multiple labels per box. default: False
    :param max_det: (int) maximum number of detections to keep. default: 300
    :returns: (list) list of filtered bounding boxes after NMS
    """
    num_classes = prediction.shape[2] - 5  # number of classes
    # create a mask of bounding boxes that have a confidence score higher than the threshold
    pred_candidates = torch.logical_and(
      prediction[..., 4] > conf_thres,
      torch.max(prediction[..., 5:], dim=-1)[0] > conf_thres
    )

    # ensure the thresholds are between 0 and 1
    assert 0 <= conf_thres <= 1
    assert 0 <= iou_thres <= 1

    # functional settings
    max_wh = 4096  # maximum box width and height
    max_nms = 30000  # maximum number of boxes put into `torchvision.ops.nms()`
    time_limit = 10.0  # quit the function when nms cost time exceed the limit time
    multi_label = multi_label & num_classes > 1  # multiple labels per box

    start = time.time()  # timer
    output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]  # list to hold the output tensor
    for image_idx, x in enumerate(prediction):  # iterate through each image in the prediction tensor
        # apply the mask to keep only the boxes whose confidence scores are higher than the threshold
        x = x[pred_candidates[image_idx]]

        if not x.shape[0]:  # if no box remains, skip to the next process
            continue

        # multiply the confidence scores by the objectness scores: conf = obj_conf * cls_conf
        x[:, 5:] = x[:, 5:] * x[:, 4:5]

        box = xywh2xyxy(x[:, :4])  # convert box format

        # if `multi_label` is enabled, keep all labels for each bounding boxes that meet the confidence threshold
        # otherwise, only keep the label with the highest confidence score
        if multi_label:
            box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
            x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), dim=1)
        else:
            conf, class_idx = x[:, 5:].max(1, keepdim=True)
            x = torch.cat([box, conf, class_idx.float()], 1)[conf.view(-1) > conf_thres]

        # filter by class, only keep the boxes whose category is in classes
        if classes is not None:
            x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

        num_box = x.shape[0]  # number of boxes
        if not num_box:  # skip if no boxes remain after filtering
            continue
        elif num_box > max_nms:
            x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

        # batched NMS
        class_offset = x[:, 5:6] * (0 if agnostic else max_wh)
        boxes, scores = x[:, :4] + class_offset, x[:, 4]
        keep_box_idx = torchvision.ops.nms(boxes, scores, iou_threshold=iou_thres)
        if keep_box_idx.shape[0] > max_det:
            keep_box_idx = keep_box_idx[:max_det]

        # keep only the bounding boxes that were not suppressed
        output[image_idx] = x[keep_box_idx]

        # time limit exceeded, log a warning message and break the loop
        if time.time() - start > time_limit:
            LOGGER.warning(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
            break  # time limit exceeded

    return output
