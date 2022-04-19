import torch
from torch import nn

from .smoke_predictor import make_smoke_predictor
from .loss import make_smoke_loss_evaluator
from .inference import make_smoke_post_processor


class SMOKEHead(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKEHead, self).__init__()

        self.cfg = cfg.clone()
        self.predictor = make_smoke_predictor(cfg, in_channels)
        self.loss_evaluator = make_smoke_loss_evaluator(cfg)
        self.post_processor = make_smoke_post_processor(cfg)

    def forward(self, features, targets=None):
        x = self.predictor(features)

        if self.training:
            loss_heatmap, loss_regression = self.loss_evaluator(x, targets)
            return {}, dict(hm_loss=loss_heatmap,
                            reg_loss=loss_regression, )
        if not self.training:
            result = self.post_processor(x, targets)

            return result, {}


class SMOKE_external_Head(nn.Module):
    def __init__(self, cfg, in_channels):
        super(SMOKE_external_Head, self).__init__()

        self.cfg = cfg.clone()
        self.predictor = make_smoke_predictor(cfg, in_channels, 'pitch_roll_head')
        self.loss_evaluator = make_smoke_loss_evaluator(cfg, 'pitch_roll_head')
        self.post_processor = make_smoke_post_processor(cfg, pred_2d=False, head_type='pitch_roll_head')
        #self.post_processor = make_smoke_post_processor(cfg)

    def forward(self, features, targets=None):
        x = self.predictor(features)
        if self.training:
            pitch_loss, roll_loss = self.loss_evaluator(x, targets)
            result = self.post_processor(x, targets)
            return result, dict(pitch_loss=pitch_loss, roll_loss=roll_loss)
        if not self.training:
            result = self.post_processor(x, targets)
            return result, {}

            





def build_smoke_head(cfg, in_channels):
    return SMOKEHead(cfg, in_channels)
def build_external_para_head(cfg, in_channels):
    return SMOKE_external_Head(cfg, in_channels)