from functools import reduce
from operator import add

import torch

from ..builder import DETECTORS
from .two_stage import TwoStageDetector
from ..correlation.correlation import Correlation
from ..plugins.gain_ref_query_feats import ExtractQueryROI
import pickle


@DETECTORS.register_module()
class FDCA(TwoStageDetector):
    def __init__(self,
                 backbone,
                 rpn_head,
                 roi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(FDCA, self).__init__(backbone=backbone,
                                    neck=neck,
                                    rpn_head=rpn_head,
                                    roi_head=roi_head,
                                    train_cfg=train_cfg,
                                    test_cfg=test_cfg,
                                    pretrained=pretrained,
                                    init_cfg=init_cfg)
        self.feat_ids = list(range(1, 17))
        bottleneck_num = [3, 4, 6, 3]
        self.bottleneck_ids = reduce(add, list(map(lambda x: list(range(x)), bottleneck_num)))
        self.layer_ids = reduce(add, [[i + 1] * x for i, x in enumerate(bottleneck_num)])
        self.stack_ids = torch.tensor(self.layer_ids).bincount()[1:5].cumsum(dim=0)[:4]
        # self.stack_ids = torch.tensor([3, 5, 8, 11])
        self.extract_ref_query_feats = ExtractQueryROI(256)

    def extract_feature(self, img):
        feats = []

        # Layer 0
        feat = self.backbone.conv1.forward(img)
        feat = self.backbone.bn1.forward(feat)
        feat = self.backbone.relu.forward(feat)
        feat = self.backbone.maxpool.forward(feat)

        # Layer 1-4
        for hid, (bid, lid) in enumerate(zip(self.bottleneck_ids, self.layer_ids)):
            res = feat
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn1.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn2.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].conv3.forward(feat)
            feat = self.backbone.__getattr__('layer%d' % lid)[bid].bn3.forward(feat)

            if bid == 0:
                res = self.backbone.__getattr__('layer%d' % lid)[bid].downsample.forward(res)

            feat += res

            # if hid + 1 in [1, 2, 3, 5, 7, 9, 11, 13, 14, 15, 16]:
            if hid + 1 in self.feat_ids:
                feats.append(feat.clone())

            feat = self.backbone.__getattr__('layer%d' % lid)[bid].relu.forward(feat)

        return feats

    def extract_ref_feature(self, ref_img):
        return self.backbone(ref_img)

    def fusion_feature(self, img):
        target_img = img[0]
        query_img = img[1]
        query_bbox = img[2]
        target_feats = self.extract_feature(target_img)
        query_feats = self.extract_feature(query_img)
        corr_feats = Correlation.multilayer_correlation(target_feats, query_feats, self.stack_ids)
        relevant_feature = self.neck(corr_feats)
        rf_feat = self.extract_ref_feature(query_img)
        ref_roi_feats = self.extract_ref_query_feats(rf_feat, query_bbox)
        return relevant_feature, ref_roi_feats

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_bboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      **kwargs):
        relevant_feature, ref_roi_feature = self.fusion_feature(img)

        losses = dict()
        accuracies = dict()

        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)

            rpn_losses, proposal_list = self.rpn_head.forward_train(
                relevant_feature,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg
            )
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        roi_losses = self.roi_head.forward_train(relevant_feature, ref_roi_feature,
                                                 img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)

        losses.update(roi_losses)
        # print(losses)
        return losses

    def simple_test(self, img, img_metas, proposals=None, rescale=False):
        """Test without augmentation."""
        assert self.with_bbox, "Bbox head must be implemented."

        relevant_feature, ref_roi_feature = self.fusion_feature(img)

        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(relevant_feature, img_metas)
        else:
            proposal_list = proposals

        return self.roi_head.simple_test(
            relevant_feature, ref_roi_feature, proposal_list, img_metas, rescale=rescale)
