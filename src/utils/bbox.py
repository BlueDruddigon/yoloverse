import torch


def dist2bbox(distances: torch.Tensor, anchor_points: torch.Tensor, bbox_format: str = 'xyxy') -> torch.Tensor:
    """Transform distance(ltrb) to bbox(xyxy or xywh)

    :param distances: distances from center point to actual bounding box
    :param anchor_points: anchor's center points coordinates
    :param bbox_format: returned bounding boxes format
    :return: bounding boxes tensor
    """
    lt, rb = torch.split(distances, 2, dim=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if bbox_format == 'xyxy':
        return torch.cat([x1y1, x2y2], dim=-1)

    c_xy = (x1y1+x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat([c_xy, wh], dim=-1)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: int) -> torch.Tensor:
    """Transform bbox(xyxy) to distance(ltrb)

    :param anchor_points: anchor's center points coordinates
    :param bbox: bounding boxes coordinates
    :param reg_max: maximum allowed value for predictions in regression branch
    :return: distances from anchor points to bounding boxes
    """
    x1y1, x2y2 = torch.split(bbox, 2, dim=-1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], dim=-1).clip(0, reg_max - 0.01)
    return dist


def xywh2xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """convert bbox from xywh to xyxy format

    :param bbox: a bounding box tensor to be converted.
    :return: converted tensor.
    """
    bbox[..., 0] = bbox[..., 0] - bbox[..., 2] * 0.5
    bbox[..., 1] = bbox[..., 1] - bbox[..., 3] * 0.5
    bbox[..., 2] = bbox[..., 0] + bbox[..., 2]
    bbox[..., 3] = bbox[..., 1] + bbox[..., 3]
    return bbox


def xyxy2xywh(bbox: torch.Tensor) -> torch.Tensor:
    """convert bbox from xyxy to xywh format

    :param bbox: a bounding box tensor to be converted.
    :return: converted tensor.
    """
    bbox[..., 0] = (bbox[..., 0] + bbox[..., 2]) / 2
    bbox[..., 1] = (bbox[..., 1] + bbox[..., 3]) / 2
    bbox[..., 2] = bbox[..., 2] - bbox[..., 0]
    bbox[..., 3] = bbox[..., 3] - bbox[..., 1]
    return bbox


def bbox_iou(bbox1: torch.Tensor, bbox2: torch.Tensor) -> torch.Tensor:
    """calculates the IoU of two bounding boxes

    :param bbox1: (Tensor) first bounding box, with coordinates (xmin, ymin, xmax, ymax) format
    :param bbox2: (Tensor) second bounding box, with same coordinates format
    :return: (Tensor) the IoU score of 2 bounding boxes
    """
    def bbox_area(bbox: torch.Tensor) -> torch.Tensor:
        """calculates the area of a bounding box

        :param bbox: (Tensor) bounding box with coordinates (xmin, ymin, xmax, ymax) format
        :returns: (Tensor) the area of the bounding box
        """
        # formula: `Area = Width * Height`
        return (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])

    area1 = bbox_area(bbox1)
    area2 = bbox_area(bbox2)

    # calculate intersection of 2 bounding boxes
    inter = (torch.min(bbox1[:, None, 2:], bbox2[:, 2:])) - torch.max(bbox1[:, None, :2], bbox2[:, :2]).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # IoU score = `inter / (area1 + area2 - inter)`
