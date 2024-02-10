import torch


def dist2bbox(distances: torch.Tensor, anchor_points: torch.Tensor, bbox_format: str = 'xyxy') -> torch.Tensor:
    """Transform distance(ltrb) to bbox(xyxy or xywh)"""
    lt, rb = torch.split(distances, 2, dim=-1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if bbox_format == 'xyxy':
        return torch.cat([x1y1, x2y2], dim=-1)

    c_xy = (x1y1+x2y2) / 2
    wh = x2y2 - x1y1
    return torch.cat([c_xy, wh], dim=-1)


def bbox2dist(anchor_points: torch.Tensor, bbox: torch.Tensor, reg_max: float):
    """Transform bbox(xyxy) to distance(ltrb)"""
    x1y1, x2y2 = torch.split(bbox, 2, dim=-1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], dim=-1).clip(0, reg_max - 0.01)
    return dist


def xywh2xyxy(bbox: torch.Tensor) -> torch.Tensor:
    """Convert bbox from xywh to xyxy format"""
    bbox[..., 0] = bbox[..., 0] - bbox[..., 2] * 0.5
    bbox[..., 1] = bbox[..., 1] - bbox[..., 3] * 0.5
    bbox[..., 2] = bbox[..., 0] + bbox[..., 2]
    bbox[..., 3] = bbox[..., 1] + bbox[..., 3]
    return bbox
