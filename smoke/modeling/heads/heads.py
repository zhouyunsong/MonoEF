import torch

from .smoke_head.smoke_head import build_smoke_head
from .smoke_head.smoke_head import build_external_para_head


def build_heads(cfg, in_channels, head_flag):
    if cfg.MODEL.SMOKE_ON:
        if head_flag==0:
            return build_external_para_head(cfg, in_channels)
        else:
            return build_smoke_head(cfg, in_channels)
