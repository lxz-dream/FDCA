import warnings
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import BaseModule, auto_fp16
from ..correlation.conv4d import CenterPivotConv4d as Conv4d

from ..builder import NECKS


@NECKS.register_module()
class FCP(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=None,
                 act_cfg=None,
                 upsample_cfg=dict(mode='nearest'),
                 init_cfg=dict(
                     type='Xavier', layer='Conv2d', distribution='uniform')):
        super(FCP, self).__init__(init_cfg)

        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.fp16_enabled = False
        self.upsample_cfg = upsample_cfg.copy()

        if end_level == -1:
            self.backone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        assert isinstance(add_extra_convs, (str, bool))
        if isinstance(add_extra_convs, str):
            assert add_extra_convs in ('on_input', 'on_lateral', 'on_output')
        elif add_extra_convs:
            if extra_convs_on_inputs:
                # TODO: deprecate `extra_convs_on_inputs`
                warnings.simplefilter('once')
                warnings.warn(
                    '"extra_convs_on_inputs" will be deprecated in v2.9.0,'
                    'Please use "add_extra_convs"', DeprecationWarning)
                self.add_extra_convs = 'on_input'
            else:
                self.add_extra_convs = 'on_output'

        def make_building_block(in_channel, out_channels, kernel_sizes, strides, group=4):
            assert len(out_channels) == len(kernel_sizes) == len(strides)

            building_block_layers = []
            for idx, (outch, ksz, stride) in enumerate(zip(out_channels, kernel_sizes, strides)):
                inch = in_channel if idx == 0 else out_channels[idx - 1]
                ksz4d = (ksz,) * 4
                str4d = (1, 1) + (stride,) * 2
                pad4d = (ksz // 2,) * 4

                building_block_layers.append(Conv4d(inch, outch, ksz4d, str4d, pad4d))
                building_block_layers.append(nn.GroupNorm(group, outch))
                building_block_layers.append(nn.ReLU(inplace=True))

            return nn.Sequential(*building_block_layers)

        self.corr_feat_squeeze_conv2 = make_building_block(3, [16, 32, 64, 128], [5, 5, 5, 3], [4, 4, 2, 1])
        self.corr_feat_squeeze_conv3 = make_building_block(4, [16, 32, 64, 128], [5, 5, 3, 3], [4, 2, 2, 1])
        self.corr_feat_squeeze_conv4 = make_building_block(6, [16, 32, 64, 128], [5, 3, 3, 3], [2, 2, 2, 1])
        self.corr_feat_squeeze_conv5 = make_building_block(3, [16, 32, 64, 128], [3, 3, 3, 3], [2, 2, 1, 1])

        # self.corr_feat_mix_convs = [
        #     make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1]),
        #     make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1]),
        #     make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1]),
        #     make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1]),
        # ]
        self.corr_feat_mix_conv2 = make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])
        self.corr_feat_mix_conv3 = make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])
        self.corr_feat_mix_conv4 = make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])
        self.corr_feat_mix_conv5 = make_building_block(128, [128, 128, 128], [3, 3, 3], [1, 1, 1])

    def upsample_target_dim(self, corr_feat, spatial_size=None):
        bsz, ch, ha, wa, hb, wb = corr_feat.size()
        corr_feat = corr_feat.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        corr_feat = F.interpolate(corr_feat, spatial_size, mode='bilinear', align_corners=True)
        o_hb, o_wb = spatial_size
        corr_feat = corr_feat.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return corr_feat

    def maxpool_target_dim(self, corr_feat):
        bsz, ch, ha, wa, hb, wb = corr_feat.size()
        corr_feat = corr_feat.permute(0, 4, 5, 1, 2, 3).contiguous().view(bsz * hb * wb, ch, ha, wa)
        corr_feat = F.max_pool2d(corr_feat, 1, stride=2)
        o_hb, o_wb = corr_feat.size()[2], corr_feat.size()[3]
        corr_feat = corr_feat.view(bsz, hb, wb, ch, o_hb, o_wb).permute(0, 3, 4, 5, 1, 2).contiguous()
        return corr_feat

    @auto_fp16()
    def forward(self, inputs):
        assert len(inputs) == len(self.in_channels)

        corr_feat_squeeze_layers = [
            self.corr_feat_squeeze_conv2(inputs[0]),
            self.corr_feat_squeeze_conv3(inputs[1]),
            self.corr_feat_squeeze_conv4(inputs[2]),
            self.corr_feat_squeeze_conv5(inputs[3])
        ]

        used_backbone_levels = len(corr_feat_squeeze_layers)
        for i in range(used_backbone_levels - 1, 0, -1):
            spatial_size = corr_feat_squeeze_layers[i - 1].size()[-4:-2]
            corr_feat_squeeze_layers[i - 1] = corr_feat_squeeze_layers[i - 1] + self.upsample_target_dim(corr_feat_squeeze_layers[i], spatial_size)

        outs = [
            self.corr_feat_mix_conv2(corr_feat_squeeze_layers[0]),
            self.corr_feat_mix_conv3(corr_feat_squeeze_layers[1]),
            self.corr_feat_mix_conv4(corr_feat_squeeze_layers[2]),
            self.corr_feat_mix_conv5(corr_feat_squeeze_layers[3])
        ]

        if self.num_outs > len(outs):
            outs.append(self.maxpool_target_dim(outs[-1]))

        for i in range(self.num_outs):
            bsz, ch, ha, wa, hb, wb = outs[i].size()
            outs[i] = outs[i].view(bsz, ch, ha, wa, -1).mean(dim=-1)

        return outs
