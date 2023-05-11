import torch


class Correlation:

    @classmethod
    def multilayer_correlation(cls, target_feats, query_feats, stack_ids):
        eps = 1e-5

        corrs = []
        for idx, (target_feat, query_feat) in enumerate(zip(target_feats, query_feats)):
            bsz, ch, hb, wb = query_feat.size()
            query_feat = query_feat.view(bsz, ch, -1)
            query_feat = query_feat / (query_feat.norm(dim=1, p=2, keepdim=True) + eps)

            bsz, ch, ha, wa = target_feat.size()
            target_feat = target_feat.view(bsz, ch, -1)
            target_feat = target_feat / (target_feat.norm(dim=1, p=2, keepdim=True) + eps)

            corr = torch.bmm(target_feat.transpose(1, 2), query_feat).view(bsz, ha, wa, hb, wb)
            corr = corr.clamp(min=0)
            corrs.append(corr)

        corr_l2 = torch.stack(corrs[0: stack_ids[0]]).transpose(0, 1).contiguous()
        corr_l3 = torch.stack(corrs[stack_ids[0]: stack_ids[1]]).transpose(0, 1).contiguous()
        corr_l4 = torch.stack(corrs[stack_ids[1]: stack_ids[2]]).transpose(0, 1).contiguous()
        corr_l5 = torch.stack(corrs[stack_ids[2]: stack_ids[3]]).transpose(0, 1).contiguous()

        return [corr_l2, corr_l3, corr_l4, corr_l5]

    @classmethod
    def roi_feature_correlation(cls, target_roi_feats, query_roi_feats):
        eps = 1e-5

        roi_num, ch, hb, wb = query_roi_feats.size()
        query_roi_feats = query_roi_feats.view(roi_num, ch, -1)
        query_roi_feats = query_roi_feats / (query_roi_feats.norm(dim=1, p=2, keepdim=True) + eps)

        roi_num, ch, ha, wa = target_roi_feats.size()
        target_roi_feats = target_roi_feats.view(roi_num, ch, -1)
        target_roi_feats = target_roi_feats / (target_roi_feats.norm(dim=1, p=2, keepdim=True) + eps)

        corr = torch.bmm(target_roi_feats.transpose(1, 2), query_roi_feats).view(roi_num, ha, wa, hb, wb)
        corr = corr.clamp(min=0)

        corr = torch.stack([corr]).transpose(0, 1)

        return corr