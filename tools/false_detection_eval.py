import pickle
import time

import sklearn as sk
from sklearn import metrics
import torch
import numpy as np
from mmdet.core.bbox.iou_calculators import bbox_overlaps


def xywh2xyxy(bbox):
    return [
        bbox[0],
        bbox[1],
        bbox[0] + bbox[2],
        bbox[1] + bbox[3],
    ]


def stable_cumsum(arr, rtol=1e-05, atol=1e-08):
    """Use high precision for cumsum and check that final value matches sum
    Parameters
    ----------
    arr : array-like
        To be cumulatively summed as flat
    rtol : float
        Relative tolerance, see ``np.allclose``
    atol : float
        Absolute tolerance, see ``np.allclose``
    """
    out = np.cumsum(arr, dtype=np.float64)
    expected = np.sum(arr, dtype=np.float64)
    if not np.allclose(out[-1], expected, rtol=rtol, atol=atol):
        raise RuntimeError('cumsum was found to be unstable: '
                           'its last element does not correspond to sum')
    return out


def fpr_and_fdr_at_recall(y_true, y_score, recall_level=0.95,
                          pos_label=None, return_index=False):
    classes = np.unique(y_true)
    if (pos_label is None and
            not (np.array_equal(classes, [0, 1]) or
                 np.array_equal(classes, [-1, 1]) or
                 np.array_equal(classes, [0]) or
                 np.array_equal(classes, [-1]) or
                 np.array_equal(classes, [1]))):
        raise ValueError("Data is not binary and pos_label is not specified")
    elif pos_label is None:
        pos_label = 1.

    # make y_true a boolean vector
    y_true = (y_true == pos_label)

    # sort scores and corresponding truth values
    desc_score_indices = np.argsort(y_score, kind="mergesort")[::-1]
    y_score = y_score[desc_score_indices]
    y_true = y_true[desc_score_indices]

    # y_score typically has many tied values. Here we extract
    # the indices associated with the distinct values. We also
    # concatenate a value for the end of the curve.
    distinct_value_indices = np.where(np.diff(y_score))[0]
    # import ipdb;
    # ipdb.set_trace()
    threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]

    # accumulate the true positives with decreasing threshold
    tps = stable_cumsum(y_true)[threshold_idxs]
    fps = 1 + threshold_idxs - tps  # add one because of zero-based indexing

    thresholds = y_score[threshold_idxs]

    recall = tps / tps[-1]
    recall_fps = fps / fps[-1]
    # breakpoint()
    ## additional code for calculating.
    if return_index:
        recall_level_fps = 1 - 0.95
        index_for_tps = threshold_idxs[np.argmin(np.abs(recall - recall_level))]
        index_for_fps = threshold_idxs[np.argmin(np.abs(recall_fps - recall_level_fps))]
        index_for_id_initial = []
        index_for_ood_initial = []
        for index in range(index_for_fps, index_for_tps + 1):
            if y_true[index] == 1:
                index_for_id_initial.append(desc_score_indices[index])
            else:
                index_for_ood_initial.append(desc_score_indices[index])
    # import ipdb;
    # ipdb.set_trace()
    ##
    last_ind = tps.searchsorted(tps[-1])
    sl = slice(last_ind, None, -1)  # [last_ind::-1]
    recall, fps, tps, thresholds = np.r_[recall[sl], 1], np.r_[fps[sl], 0], np.r_[tps[sl], 0], thresholds[sl]

    cutoff = np.argmin(np.abs(recall - recall_level))

    # 8.868, ours
    # 5.772, vanilla
    # 5.478, vanilla 18000
    # 6.018, oe
    # 102707,
    # 632
    # 5992
    # breakpoint()
    if return_index:
        return fps[cutoff] / (np.sum(np.logical_not(y_true))), index_for_id_initial, index_for_ood_initial
    else:
        return fps[cutoff] / (np.sum(np.logical_not(y_true)))
    # , fps[cutoff]/(fps[cutoff] + tps[cutoff])


def get_measures(_pos, _neg, recall_level=0.95, return_index=False, plot=False):
    _pos = torch.tensor(_pos)
    _neg = torch.tensor(_neg)
    pos = np.array(_pos[:]).reshape((-1, 1))
    neg = np.array(_neg[:]).reshape((-1, 1))
    examples = np.squeeze(np.vstack((pos, neg)))
    labels = np.zeros(len(examples), dtype=np.int32)
    labels[:len(pos)] += 1

    auroc = metrics.roc_auc_score(labels, examples)
    if plot:
        # breakpoint()
        import matplotlib.pyplot as plt
        fpr1, tpr1, thresholds = metrics.roc_curve(labels, examples, pos_label=1)
        fig, ax = plt.subplots(figsize=(10, 8))
        ax.plot(fpr1, tpr1, linewidth=2,
                label='10000_1')
        ax.plot([0, 1], [0, 1], linestyle='--', color='grey')
        plt.legend(fontsize=12)
        plt.savefig('10000_1.jpg', dpi=250)
    aupr = metrics.average_precision_score(labels, examples)
    if return_index:
        fpr, index_id, index_ood = fpr_and_fdr_at_recall(labels, examples, recall_level, return_index=return_index)
        return auroc, aupr, fpr, index_id, index_ood
    else:
        fpr = fpr_and_fdr_at_recall(labels, examples, recall_level)
        return auroc, aupr, fpr


def print_measures(auroc, aupr, fpr, method_name='Ours', recall_level=0.95):
    print('\t\t\t\t' + method_name)
    print('  FPR{:d} AUROC AUPR'.format(int(100 * recall_level)))
    print('& {:.2f} & {:.2f} & {:.2f}'.format(100 * fpr, 100 * auroc, 100 * aupr))
    # print('FPR{:d}:\t\t\t{:.2f}'.format(int(100 * recall_level), 100 * fpr))
    # print('AUROC: \t\t\t{:.2f}'.format(100 * auroc))
    # print('AUPR:  \t\t\t{:.2f}'.format(100 * aupr))


with open("/root/autodl-tmp/FCP/intermediate_results.pkl", 'rb') as file:  # 读取pkl文件数据
    intermediate_data = pickle.load(file, encoding='bytes')

    print(len(intermediate_data["gts"]))
    print(len(intermediate_data["dts"]))

    data = {}
    for i in range(len(intermediate_data["gts"])):
        index = str(intermediate_data["gts"][i]["image_id"]) + "_" + str(intermediate_data["gts"][i]["category_id"])
        if index not in data.keys():
            data[index] = {}
            data[index]["gt_boxes"] = {}
            data[index]["gt_boxes"]["boxes"] = []
            data[index]["gt_boxes"]["category"] = set()
            data[index]["gt_boxes"]["boxes"].append(xywh2xyxy(intermediate_data["gts"][i]["bbox"]))
            data[index]["gt_boxes"]["category"].add(intermediate_data["gts"][i]["category_id"])
        else:
            data[index]["gt_boxes"]["boxes"].append(xywh2xyxy(intermediate_data["gts"][i]["bbox"]))
            data[index]["gt_boxes"]["category"].add(intermediate_data["gts"][i]["category_id"])
    for i in range(len(intermediate_data["dts"])):
        index = str(intermediate_data["dts"][i]["image_id"]) + "_" + str(intermediate_data["dts"][i]["category_id"])
        if index not in data.keys():
            print("error: det_boxes does not have a corresponding real box")
            break
        if "det_boxes" not in data[index].keys():
            data[index]["det_boxes"] = {}
            data[index]["det_boxes"]["boxes"] = []
            data[index]["det_boxes"]["category"] = set()
            data[index]["det_boxes"]["scores"] = []
            data[index]["det_boxes"]["boxes"].append(xywh2xyxy(intermediate_data["dts"][i]["bbox"]))
            data[index]["det_boxes"]["category"].add(intermediate_data["dts"][i]["category_id"])
            data[index]["det_boxes"]["scores"].append(intermediate_data["dts"][i]["score"])
        else:
            data[index]["det_boxes"]["boxes"].append(xywh2xyxy(intermediate_data["dts"][i]["bbox"]))
            data[index]["det_boxes"]["category"].add(intermediate_data["dts"][i]["category_id"])
            data[index]["det_boxes"]["scores"].append(intermediate_data["dts"][i]["score"])

    pos = []
    neg = []

    for image_index in data:
        if "det_boxes" in data[image_index]:
            box_det = torch.tensor(data[image_index]["det_boxes"]["boxes"])
            box_gt = torch.tensor(data[image_index]["gt_boxes"]["boxes"])
            ious = bbox_overlaps(box_det, box_gt)
            ious = ious.max(dim=1).values
            for i in range(ious.shape[0]):
                if ious[i] >= 0.6 and data[image_index]["det_boxes"]["scores"][i] >= 0.5:
                    pos.append(data[image_index]["det_boxes"]["scores"][i])
                elif ious[i] <= 0.3 and data[image_index]["det_boxes"]["scores"][i] >= 0.5:
                    neg.append(data[image_index]["det_boxes"]["scores"][i])

    measures = get_measures(pos, neg, plot=False)
    print_measures(measures[0], measures[1], measures[2])