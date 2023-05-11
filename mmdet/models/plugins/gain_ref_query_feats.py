from torchvision.ops import roi_align
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import xavier_init

class ExtractQueryROI(nn.Module):
    def __init__(self,
                 out_channels):
        super(ExtractQueryROI, self).__init__()
        assert out_channels == 256

        self.unify_channel = nn.ModuleList()
        for i in range(4):
            self.unify_channel.append(nn.Conv2d(256 * 2 ** i, 256, 1))

        self.reduce_channel = nn.Conv2d(256, 128, 3, padding=1)

        self.init_weights()

    def init_weights(self, std=0.01):
        xavier_init(self.reduce_channel, distribution='uniform')

    def gain_ref_query_feats(self, rf_feat, rf_bbox):
        roi_feats = []
        for i in range(rf_bbox.shape[0]):
            roi_feat = roi_align(rf_feat[i].unsqueeze(0), [rf_bbox[i] / 4], [7, 7])
            roi_feats.append(roi_feat)
        roi_feats = torch.cat(roi_feats, dim=0)
        return roi_feats

    def forward(self, rf_feat, rf_bbox):
        assert len(rf_feat) == 4
        rf_feat_list = list(rf_feat)
        for i in range(len(rf_feat_list)):
            rf_feat_list[i] = self.unify_channel[i](rf_feat_list[i])
        rf_feat_list[0] = rf_feat_list[0].unsqueeze(1)
        for i in range(1, len(rf_feat_list)):
            rf_feat_list[i] = F.interpolate(rf_feat_list[i], scale_factor=2 ** i, mode='nearest')
            rf_feat_list[i] = rf_feat_list[i].unsqueeze(1)
        feats = torch.cat(rf_feat_list, dim=1)
        feats = feats.mean(dim=1)
        feats = self.reduce_channel(feats)
        return self.gain_ref_query_feats(feats, rf_bbox)
