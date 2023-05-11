import torch
import torch.nn as nn

from mmcv.cnn import xavier_init
import mmcv
import torch.nn.functional as F
import numpy as np
import cv2
from ..correlation.correlation import Correlation
from ..correlation.conv4d import CenterPivotConv4d as Conv4d

# cbam的通道注意力部分
class channel_attention(nn.Module):
    def __init__(self, channel, ratio = 16):
        super(channel_attention, self).__init__()
        # maxpooling and avg pooling
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # two times:MLP
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // ratio, False),
            nn.ReLU(),
            nn.Linear(channel // ratio, channel, False),
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        b, c, w, h = x.size()
        # 先分别进行最大池化和平均池化
        max_pool_out = self.max_pool(x).view([b, c])
        avg_pool_out = self.avg_pool(x).view([b, c])
        
        # 再送入共享全连接层
        max_fc_out = self.fc(max_pool_out)
        avg_fc_out = self.fc(avg_pool_out)
        
        # 再进行相加操作，相加结果再取一个sigmoid
        out = max_fc_out + avg_fc_out
        out = self.sigmoid(out).view([b,c,1,1])
        
        return out * x

    
class Cbam(nn.Module):
    def __init__(self, channel, ratio=16, kernel_size=3):
        super(Cbam, self).__init__()
        self.channel_attention = channel_attention(channel, ratio)
        
    def forward(self, x):
        x = self.channel_attention(x)
        return x


class FDA(nn.Module):

    def __init__(self,
                 metric_module):
        super(FDA, self).__init__()

        self.module_in_channel = metric_module["in_channel"]
        self.module_out_channel = metric_module["out_channel"]

        assert self.module_in_channel == 128 and self.module_out_channel == 192
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.conv1 = nn.Conv2d(self.module_in_channel, self.module_in_channel // 2, 1, 1)
        self.metric = nn.Conv2d(self.module_in_channel*2, self.module_out_channel, 1, 1)
        self.attention_channel = self.module_in_channel

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

        self.corr_feat_squeeze_conv = make_building_block(1, [16, 64, 128], [3, 3, 3], [2, 2, 1])

        self.attetion = Cbam(self.attention_channel * 2)
  

    def forward(self, x, ref):

        contrastive_feat = (self.avg(ref) - x).abs()
        contrastive_feat = self.conv1(contrastive_feat)


        corr = Correlation.roi_feature_correlation(x, ref)
        corr = self.corr_feat_squeeze_conv(corr)
        roi_num, ch, ha, wa, hb, wb = corr.size()
        corr = corr.view(roi_num, ch, ha, wa, -1).mean(dim=-1)
        attention_feat = (corr - x).abs()
        attention_feat = self.conv1(attention_feat)

        corr_feat = self.conv1(corr)
        contrastive_and_corr_feat = self.conv1(torch.cat([corr_feat,contrastive_feat],1))

        output = torch.cat([contrastive_and_corr_feat, attention_feat, x], 1)
        
        output = self.attetion(output)
        output = self.metric(output)

        return output
