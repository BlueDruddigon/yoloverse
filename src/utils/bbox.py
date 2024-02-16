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
