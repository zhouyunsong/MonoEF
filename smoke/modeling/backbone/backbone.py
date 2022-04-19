from collections import OrderedDict

from torch import nn

from smoke.modeling import registry
from . import dla

@registry.BACKBONES.register("DLA-34-DCN")
def build_dla_backbone(cfg):
    body = dla.DLA(cfg)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.BACKBONE.BACKBONE_OUT_CHANNELS
    return model

@registry.TRANSFER.register("DLA-34-DCN")
def build_dla_transfer(cfg):
    body = dla.DLA_T(cfg, out_channel=64)
    model = nn.Sequential(OrderedDict([("body", body)]))
    model.out_channels = cfg.MODEL.TRANSFER.BACKBONE_OUT_CHANNELS
    return model


def build_backbone(cfg):
    assert cfg.MODEL.BACKBONE.CONV_BODY in registry.BACKBONES, \
        "cfg.MODEL.BACKBONE.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.BACKBONE.CONV_BODY
        )
    return registry.BACKBONES[cfg.MODEL.BACKBONE.CONV_BODY](cfg)

def build_transfer(cfg):
    assert cfg.MODEL.TRANSFER.CONV_BODY in registry.TRANSFER, \
        "cfg.MODEL.TRANSFER.CONV_BODY: {} are not registered in registry".format(
            cfg.MODEL.TRANSFER.CONV_BODY
        )
    return registry.TRANSFER[cfg.MODEL.TRANSFER.CONV_BODY](cfg)
