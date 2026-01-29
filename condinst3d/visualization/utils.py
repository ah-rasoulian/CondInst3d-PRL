from torch import Tensor
import numpy as np
from skimage.segmentation import relabel_sequential
import skimage.measure as measure


def get_stats(pairwise_iou: Tensor, scores: Tensor | None = None):
    pairwise_iou = pairwise_iou.cpu()
    stats = {"y_true": {}, "y_pred": {}}
    for gt_idx in range(pairwise_iou.size(1)):
        per_gt_iou = pairwise_iou[:, gt_idx]
        if sum(per_gt_iou) == 0:
            stats["y_true"][gt_idx + 1] = {'type': 'FN'}

    for pred_idx in range(pairwise_iou.size(0)):
        per_pred_iou = pairwise_iou[pred_idx]
        s = {}
        if scores is not None:
            s['score'] = scores[pred_idx].item()
        if sum(per_pred_iou) == 0:
            s = s | {'type': 'FP'}
            stats["y_pred"][pred_idx + 1] = s
        else:
            tp_indices = np.nonzero(per_pred_iou)[0]
            tp_indices = tp_indices.reshape(-1)
            s = s | {'type': 'TP', 'regions': tp_indices + 1, 'iou': per_pred_iou[tp_indices]}
            stats["y_pred"][pred_idx + 1] = s
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
