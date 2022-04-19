import torch
from torch import nn
from torch.nn import functional as F

from smoke.utils.registry import Registry
from smoke.modeling import registry
from smoke.layers.utils import sigmoid_hm
from smoke.modeling.make_layers import group_norm
from smoke.modeling.make_layers import _fill_fc_weights

_HEAD_NORM_SPECS = Registry({
    "BN": nn.BatchNorm2d,
    "GN": group_norm,
})


def get_channel_spec(reg_channels, name):
    if name == "dim":
        s = sum(reg_channels[:2])
        e = sum(reg_channels[:3])
    elif name == "ori":
        s = sum(reg_channels[:3])
        e = sum(reg_channels)

    return slice(s, e, 1)


@registry.SMOKE_PREDICTOR.register("SMOKEPredictor")
class SMOKEPredictor(nn.Module):
    def __init__(self, cfg, in_channels, head_type):
        super(SMOKEPredictor, self).__init__()

        classes = len(cfg.DATASETS.DETECT_CLASSES)
        regression = cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
        regression_channels = cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL
        head_conv = cfg.MODEL.SMOKE_HEAD.NUM_CHANNEL
        norm_func = _HEAD_NORM_SPECS[cfg.MODEL.SMOKE_HEAD.USE_NORMALIZATION]
        if not cfg.MODEL.SMOKE_HEAD.USE_P:
            assert sum(regression_channels) == regression, \
                "the sum of {} must be equal to regression channel of {}".format(
                    cfg.MODEL.SMOKE_HEAD.REGRESSION_CHANNEL, cfg.MODEL.SMOKE_HEAD.REGRESSION_HEADS
                )

        self.dim_channel = get_channel_spec(regression_channels, name="dim")
        self.ori_channel = get_channel_spec(regression_channels, name="ori")
        self.P_channel = 1
        self.head_type = head_type

        if self.head_type == 'det_head':
            self.class_head = nn.Sequential(
                nn.Conv2d(in_channels,
                        head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True),

                norm_func(head_conv),

                nn.ReLU(inplace=True),

                nn.Conv2d(head_conv,
                        classes,
                        kernel_size=1,
                        padding=1 // 2,
                        bias=True)
            )

            # todo: what is datafill here
            self.class_head[-1].bias.data.fill_(-2.19)

            self.regression_head = nn.Sequential(
                nn.Conv2d(in_channels,
                        head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True),

                norm_func(head_conv),

                nn.ReLU(inplace=True),

                nn.Conv2d(head_conv,
                        regression,
                        kernel_size=1,
                        padding=1 // 2,
                        bias=True)
            )
            _fill_fc_weights(self.regression_head)

        if self.head_type == 'pitch_roll_head':
            self.P_matrix_head = nn.Sequential(
                nn.Conv2d(in_channels,
                        head_conv,
                        kernel_size=3,
                        padding=1,
                        bias=True),

                norm_func(head_conv),

                nn.ReLU(inplace=True),

                nn.Conv2d(head_conv,
                        2,
                        kernel_size=1,
                        padding=1 // 2,
                        bias=True)
            )
            self.P_liner_pitch = nn.Sequential()
            self.P_liner_roll = nn.Sequential()
            inplanes = 96*320
            fc_out_channels = [96*16,96,12,1]
            num_fc = 4
            for i in range(num_fc):
                #self.P_liner.add_module(f'{i}', nn.Sequential(nn.Linear(inplanes, fc_out_channels[i]), nn.ReLU(inplace=True)))
                #self.P_liner.add_module(f'{i}', nn.Sequential(nn.ReLU(inplace=True), nn.Linear(inplanes, fc_out_channels[i])))
                self.P_liner_pitch.add_module(f'{i}', nn.Linear(inplanes, fc_out_channels[i]))
                self.P_liner_roll.add_module(f'{i}', nn.Linear(inplanes, fc_out_channels[i]))
                inplanes = fc_out_channels[i]
        
            _fill_fc_weights(self.P_matrix_head)
            _fill_fc_weights(self.P_liner_pitch)
            _fill_fc_weights(self.P_liner_roll)

    def forward(self, features):
        if self.head_type=='det_head':
            head_class = self.class_head(features)
            head_regression = self.regression_head(features)
            head_class = sigmoid_hm(head_class)
            # (N, C, H, W)
            offset_dims = head_regression[:, self.dim_channel, ...].clone()
            head_regression[:, self.dim_channel, ...] = torch.sigmoid(offset_dims) - 0.5

            vector_ori = head_regression[:, self.ori_channel, ...].clone()
            head_regression[:, self.ori_channel, ...] = F.normalize(vector_ori)

            return [head_class, head_regression]
        if self.head_type=='pitch_roll_head':
            head_pitch_roll = self.P_matrix_head(features)
            head_pitch = self.P_liner_pitch(head_pitch_roll.view(-1,2,96*320)[:,0,...])
            head_roll = self.P_liner_roll(head_pitch_roll.view(-1,2,96*320)[:,1,...])
            head_pitch_roll = torch.cat((head_pitch, head_roll), dim = -1)
            return [head_pitch_roll]


def make_smoke_predictor(cfg, in_channels, head_type='det_head'):
    func = registry.SMOKE_PREDICTOR[
        cfg.MODEL.SMOKE_HEAD.PREDICTOR
    ]
    return func(cfg, in_channels, head_type)
