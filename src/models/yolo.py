import math
from typing import List, Tuple, Union

import torch
import torch.nn as nn
from typing_extensions import Self

from src.layers.common import get_block
from src.utils.config import Config
from src.utils.torch_utils import initialize_weights

from .backbone.efficientrep import *
from .neck.reppan import *


class Model(nn.Module):
    """YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep backbone, Rep-PAN and
    Efficient Decoupled Head.
    """
    export = False

    def __init__(self, config: Config, channels: int = 3, num_classes: int = 80) -> None:
        super().__init__()
        # build network
        num_layers = config.model.head.num_layers
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes, num_layers)

        # init detect head
        self.strides = self.detect.strides
        self.detect.initialize_biases()

        # init weights
        initialize_weights(self)

    def forward(self, x: torch.Tensor) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        x = self.backbone(x)
        x = self.neck(x)
        if not export_mode:
            feat_maps = []
            feat_maps.extend(x)
        x = self.detect(x)
        return x if export_mode else (x, feat_maps)

    def _apply(self, fn: callable) -> Self:
        self = super()._apply(fn)
        self.detect.strides = fn(self.detect.strides)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self


def make_divisible(x: int, divisor: int) -> int:
    return math.ceil(x / divisor) * divisor


def build_network(config: Config, channels: int, num_classes: int, num_layers: int):
    depth_mult = config.model.depth_multiple
    width_mult = config.model.width_multiple
    num_repeats_backbone = config.model.backbone.num_repeats
    channels_list_backbone = config.model.backbone.out_channels
    fuse_p2 = config.model.backbone.get('fuse_p2')
    cspsppf = config.model.backbone.get('cspsppf')
    num_repeats_neck = config.model.neck.num_repeats
    channels_list_neck = config.model.neck.out_channels
    use_dfl = config.model.head.use_dfl
    reg_max = config.model.head.reg_max
    num_repeats = [max(round(i * depth_mult), 1) if i > 1 else i for i in (num_repeats_backbone + num_repeats_neck)]
    channels_list = [make_divisible(i * width_mult, divisor=8) for i in (channels_list_backbone + channels_list_neck)]

    block = get_block(config.training_mode)
    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    if 'CSP' in config.model.backbone.type:
        if 'stage_block_type' in config.model.backbone:
            stage_block_type = config.model.backbone.stage_block_type
        else:
            stage_block_type = 'BepC3'  # default

        backbone = BACKBONE(
          in_channels=channels,
          channels_list=channels_list,
          num_repeats=num_repeats,
          block=block,
          csp_ratio=config.model.backbone.csp_ratio,
          fuse_p2=fuse_p2,
          cspsppf=cspsppf,
          stage_block_type=stage_block_type
        )

        neck = NECK(
          channels_list=channels_list,
          num_repeats=num_repeats,
          block=block,
          csp_ratio=config.model.neck.csp_ratio,
          stage_block_type=stage_block_type
        )
    else:
        backbone = BACKBONE(
          in_channels=channels,
          channels_list=channels_list,
          num_repeats=num_repeats,
          block=block,
          fuse_p2=fuse_p2,
          cspsppf=cspsppf
        )

        neck = NECK(channels_list=channels_list, num_repeats=num_repeats, block=block)

    from .head.effidehead import EffiDeHead, build_effidehead_layers
    head_layers = build_effidehead_layers(channels_list, 1, num_classes, reg_max=reg_max, num_layers=num_layers)
    head = EffiDeHead(num_classes, num_layers, head_layers=head_layers, use_dfl=use_dfl)

    return backbone, neck, head


def build_model(cfg: Config, num_classes: int, device: Union[str, torch.device]) -> Model:
    return Model(cfg, channels=3, num_classes=num_classes).to(device)
