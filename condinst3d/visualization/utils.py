from torch import Tensor
import torch
import numpy as np
from skimage.segmentation import relabel_sequential
import skimage.measure as measure


def get_stats(pairwise_iou: Tensor, y_true_ids: Tensor, y_pred_ids: Tensor, scores: Tensor | None = None):
    """
    pairwise_iou: [K,G] where row k corresponds to y_pred_ids[k],
                  col g corresponds to y_true_ids[g]
    y_true_ids: [G] label values present in y_true (excluding 0)
    y_pred_ids: [K] label values present in y_pred (excluding 0)
    scores: [K] aligned with rows (same order as y_pred_ids)
    """
    pairwise_iou = pairwise_iou.detach().cpu()
    y_true_ids = y_true_ids.detach().cpu().tolist()
    y_pred_ids = y_pred_ids.detach().cpu().tolist()

    stats = {"y_true": {}, "y_pred": {}}

    # FN: GT with no overlap with any pred
    for g, gt_id in enumerate(y_true_ids):
        per_gt = pairwise_iou[:, g]
        if float(per_gt.sum()) == 0.0:
            stats["y_true"][int(gt_id)] = {"type": "FN"}

    # FP/TP per prediction
    for k, pred_id in enumerate(y_pred_ids):
        per_pred = pairwise_iou[k]
        s = {}
        if scores is not None:
            s["score"] = float(scores[k].item())

        if float(per_pred.sum()) == 0.0:
            stats["y_pred"][int(pred_id)] = {**s, "type": "FP"}
        else:
            tp_cols = torch.nonzero(per_pred > 0, as_tuple=False).view(-1).cpu().numpy()
            regions = [int(y_true_ids[c]) for c in tp_cols]
            stats["y_pred"][int(pred_id)] = {
                **s,
                "type": "TP",
                "regions": regions,          # actual GT label ids
                "iou": per_pred[tp_cols],    # tensor on CPU
            }

    return stats


def count_error_type(tp_overlap_size=1, stats=None, y_true=None, y_pred=None):
    """
    Returns how many FN, FP and TP there are when compared to reference
    """
    if stats is None:
        stats = error_type_region_wise(
            y_true, y_pred, tp_overlap_size=tp_overlap_size)
    summary = {'FN': len(stats['y_true']),
               'FP': len([i for i in stats['y_pred'].values() if
                          i['type'] == 'FP']),
               'TP': len([i for i in stats['y_pred'].values() if
                          i['type'] == 'TP'])
               }
    return summary

def error_type_region_wise(y_true, y_pred, tp_overlap_size=1):
    """
    Returns a dictionary with tagged error type for each regions from both reference and this label.
    """
    inter, union = intersection_union(y_true, y_pred)
    stats = {'y_true': dict(), 'y_pred': dict()}
    ref_region_not_matched = set()

    for k_s, v_s in inter.items():
        # Tuple reference labelname and size of intersection
        len_inter = [(k, len(v)) for k, v in v_s.items()]
        # Filter regions which satisfy TRUE POSITIVe criteria
        bool_tp = np.array([i[1] for i in len_inter]) >= tp_overlap_size

        # The region is a TP, find matches in the reference
        if np.any(bool_tp):
            for i, v in enumerate(bool_tp):
                name = len_inter[i][0]
                if v is False:
                    continue
                iou = len_inter[i][1] / len(union[k_s][name])
                if k_s in stats['y_pred']:
                    stats['y_pred'][k_s]['regions'].append(name)
                    stats['y_pred'][k_s]['iou'].append(iou)
                else:
                    stats['y_pred'][k_s] = {'type': 'TP', 'regions': [name], 'iou': [iou]}
                # Remove ref region as a potention FN
                if name in ref_region_not_matched:
                    ref_region_not_matched.remove(name)
        # The region is a FP, add all regions from ref as potential FN
        else:
            ref_region_not_matched.update(list(v_s.keys()))
            stats['y_pred'][k_s] = {'type': 'FP'}

    # Label all remaining regions from ref as FN
    for i in ref_region_not_matched:
        stats['y_true'][i] = {'type': 'FN'}

    for i in regions_id(y_true):
        tp_flag = False
        for v in stats['y_pred'].values():
            if v['type'] == 'TP':
                if i in v['regions']:
                    tp_flag = True
                    break
        if not tp_flag:
            stats['y_true'][i] = {'type': 'FN'}
    return stats

def intersection_union(y_true, y_pred):
    """
    Returns the indices of intersection with a reference label.
    """
    idx_s = regions_indices(y_pred)
    idx_r = regions_indices(y_true)
    stats_intersection = dict()
    stats_union = dict()
    for k_s, v_s in idx_s.items():
        tr_s = set(tuple(i) for i in np.transpose(v_s))
        stats_intersection[k_s] = dict()
        stats_union[k_s] = dict()
        for k_r, v_r in idx_r.items():
            tr_r = set(tuple(i) for i in np.transpose(v_r))
            inter = tr_s.intersection(tr_r)
            if len(inter):
                union = tr_s.union(tr_r)
                stats_intersection[k_s][k_r] = inter
                stats_union[k_s][k_r] = union
    return stats_intersection, stats_union

def regions_indices(label):
    """
    Returns a dictionary with the coordinates for each regions

    :rtype dict
    """
    return dict((i, np.where(label == i)) for i in regions_id(label))


def regions_id(label):
    """
    Returns a list of valid regions ID.

    :rtype: list of int
    """
    key = list(np.unique(label))
    if key[0] == 0:
        del key[0]
    return key


def simple_binary_to_gray(binary):
    cc = measure.label(binary)
    cc, _, _ = relabel_sequential(cc)
    return cc.astype(np.int16)
