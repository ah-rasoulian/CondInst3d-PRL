from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Tuple, Union
from dataclasses import dataclass
import json
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from sklearn.cluster import KMeans


Pathlike = Union[str, Path]


@dataclass
class LesionBox:
    case_id: str
    instance_id: int
    size_vox: np.ndarray   # [sx, sy, sz] in voxels
    size_mm: np.ndarray    # [sx, sy, sz] in mm


def get_case_ids_from_labels_dir(labels_dir: Path) -> List[str]:
    return sorted([p.name[:-7] for p in labels_dir.glob("*.nii.gz")])


def read_label(label_path: Path) -> sitk.Image:
    return sitk.ReadImage(str(label_path))


def extract_instance_boxes(label_path: Path) -> List[LesionBox]:
    """
    Extract one bbox per nonzero instance ID.
    Size is bbox side length in voxels and mm.
    Uses SimpleITK spacing, but reorders to array axis order [x, y, z] matching bbox dims.
    """
    img = read_label(label_path)
    arr = sitk.GetArrayFromImage(img)  # [z, y, x]
    spacing_xyz = np.array(img.GetSpacing(), dtype=np.float64)  # [x, y, z]

    case_id = label_path.name[:-7]
    instance_ids = np.unique(arr)
    instance_ids = instance_ids[instance_ids > 0]

    boxes: List[LesionBox] = []

    for inst_id in instance_ids:
        zz, yy, xx = np.where(arr == inst_id)
        if len(zz) == 0:
            continue

        zmin, zmax = zz.min(), zz.max()
        ymin, ymax = yy.min(), yy.max()
        xmin, xmax = xx.min(), xx.max()

        # bbox side length in voxels, converted to [x, y, z]
        size_vox = np.array([
            xmax - xmin + 1,
            ymax - ymin + 1,
            zmax - zmin + 1,
        ], dtype=np.float64)

        size_mm = size_vox * spacing_xyz

        boxes.append(
            LesionBox(
                case_id=case_id,
                instance_id=int(inst_id),
                size_vox=size_vox,
                size_mm=size_mm,
            )
        )

    return boxes


def gather_all_boxes(labels_tr: Path) -> List[LesionBox]:
    all_boxes: List[LesionBox] = []
    for label_path in tqdm(sorted(labels_tr.glob("*.nii.gz")), desc="Gathering all boxes"):
        all_boxes.extend(extract_instance_boxes(label_path))
    return all_boxes


def box_iou_same_center_3d(box_a: np.ndarray, box_b: np.ndarray) -> float:
    """
    IoU between two axis-aligned boxes centered at same point,
    represented only by side lengths [sx, sy, sz].
    """
    inter = np.minimum(box_a, box_b).prod()
    union = box_a.prod() + box_b.prod() - inter
    if union <= 0:
        return 0.0
    return float(inter / union)


def best_anchor_ious(gt_boxes_mm: np.ndarray, anchors_mm: np.ndarray) -> np.ndarray:
    """
    gt_boxes_mm: [N, 3]
    anchors_mm: [K, 3]
    returns: [N] best IoU per gt box
    """
    out = np.zeros(len(gt_boxes_mm), dtype=np.float64)
    for i, gt in enumerate(gt_boxes_mm):
        out[i] = max(box_iou_same_center_3d(gt, a) for a in anchors_mm)
    return out


def representative_anchor_from_cluster(
    cluster_boxes_mm: np.ndarray,
    method: str = "median",
) -> np.ndarray:
    """
    One anchor shape for one cluster.
    Robust choices:
      - median
      - q75
      - geometric_mean
    """
    if method == "median":
        return np.median(cluster_boxes_mm, axis=0)
    elif method == "q75":
        return np.quantile(cluster_boxes_mm, 0.75, axis=0)
    elif method == "geometric_mean":
        return np.exp(np.mean(np.log(np.clip(cluster_boxes_mm, 1e-6, None)), axis=0))
    else:
        raise ValueError(f"Unknown method: {method}")


def monotonic_sort_anchors(anchors_mm: np.ndarray) -> np.ndarray:
    """
    Sort anchors from small to large using physical volume.
    """
    vols = np.prod(anchors_mm, axis=1)
    order = np.argsort(vols)
    return anchors_mm[order]


def enforce_monotonic_growth(
    anchors_mm: np.ndarray,
    min_growth: float = 1.10,
) -> np.ndarray:
    """
    Ensure each next anchor is larger than previous one in every axis,
    at least by min_growth overall where needed.
    """
    anchors = anchors_mm.copy()
    for i in range(1, len(anchors)):
        prev = anchors[i - 1]
        cur = anchors[i]

        # make each axis non-decreasing
        cur = np.maximum(cur, prev)

        # ensure overall growth in volume/scale
        prev_vol = prev.prod()
        cur_vol = cur.prod()
        target_vol = prev_vol * min_growth
        if cur_vol < target_vol:
            scale = (target_vol / max(cur_vol, 1e-8)) ** (1.0 / 3.0)
            cur = cur * scale

        anchors[i] = cur
    return anchors


def cluster_anchors_from_boxes(
    gt_boxes_mm: np.ndarray,
    k: int,
    anchor_stat: str = "median",
    random_state: int = 0,
) -> np.ndarray:
    """
    Cluster lesions by 1D scale surrogate, then derive one anisotropic anchor per cluster.
    """
    # 1D size surrogate for ordering levels
    log_vol = np.log(np.prod(gt_boxes_mm, axis=1)).reshape(-1, 1)

    km = KMeans(n_clusters=k, n_init=20, random_state=random_state)
    labels = km.fit_predict(log_vol)

    anchors = []
    for c in range(k):
        cluster_boxes = gt_boxes_mm[labels == c]
        anchor = representative_anchor_from_cluster(cluster_boxes, method=anchor_stat)
        anchors.append(anchor)

    anchors = np.stack(anchors, axis=0)
    anchors = monotonic_sort_anchors(anchors)
    anchors = enforce_monotonic_growth(anchors, min_growth=1.10)
    return anchors


def round_anchor_to_feature_friendly_sizes(
    anchor_mm: np.ndarray,
    spacing_xyz: Tuple[float, float, float] = (0.8, 0.8, 2.0),
    round_to_vox: bool = True,
    even_vox: bool = True,
) -> Dict[str, np.ndarray]:
    """
    Convert anchor mm to voxel-aligned sizes, optionally rounded to even voxel counts.
    """
    spacing_xyz = np.array(spacing_xyz, dtype=np.float64)
    vox = anchor_mm / spacing_xyz

    if round_to_vox:
        vox = np.round(vox).astype(int)
        vox = np.maximum(vox, 1)

        if even_vox:
            vox = np.where(vox % 2 == 1, vox + 1, vox)

    mm = vox * spacing_xyz
    return {"vox": vox, "mm": mm}


def score_anchor_set(
    gt_boxes_mm: np.ndarray,
    anchors_mm: np.ndarray,
    k_penalty: float = 0.01,
) -> Dict[str, float]:
    """
    Score anchor set by lesion coverage and small penalty for more levels.
    """
    best_ious = best_anchor_ious(gt_boxes_mm, anchors_mm)
    mean_iou = float(best_ious.mean())
    median_iou = float(np.median(best_ious))
    recall_025 = float((best_ious >= 0.25).mean())
    recall_050 = float((best_ious >= 0.50).mean())

    # Main score: prioritize mean IoU, modestly favor fewer levels
    score = mean_iou - k_penalty * len(anchors_mm)

    return {
        "score": score,
        "mean_best_iou": mean_iou,
        "median_best_iou": median_iou,
        "recall_iou_ge_0.25": recall_025,
        "recall_iou_ge_0.50": recall_050,
    }


def find_best_anchors(
    labels_tr: Pathlike,
    spacing_xyz: Tuple[float, float, float] = (0.8, 0.8, 2.0),
    min_levels: int = 2,
    max_levels: int = 8,
    anchor_stat: str = "median",
    random_state: int = 0,
    k_penalty: float = 0.01,
) -> Dict:
    """
    Determine the best number of anchor levels and one anchor per level from the full train set.
    """
    labels_tr = Path(labels_tr)
    boxes = gather_all_boxes(labels_tr)

    if len(boxes) == 0:
        raise ValueError(f"No lesion instances found in {labels_tr}")

    gt_boxes_mm = np.stack([b.size_mm for b in boxes], axis=0)

    results = []
    best = None

    for k in range(min_levels, max_levels + 1):
        anchors_mm = cluster_anchors_from_boxes(
            gt_boxes_mm=gt_boxes_mm,
            k=k,
            anchor_stat=anchor_stat,
            random_state=random_state,
        )

        rounded = [round_anchor_to_feature_friendly_sizes(a, spacing_xyz=spacing_xyz) for a in anchors_mm]
        rounded_mm = np.stack([r["mm"] for r in rounded], axis=0)
        rounded_vox = np.stack([r["vox"] for r in rounded], axis=0)

        metrics = score_anchor_set(gt_boxes_mm, rounded_mm, k_penalty=k_penalty)

        item = {
            "num_levels": k,
            "anchors_mm": rounded_mm,
            "anchors_vox": rounded_vox,
            **metrics,
        }
        results.append(item)

        if best is None or item["score"] > best["score"]:
            best = item

    assert best is not None

    summary = {
        "num_instances": len(boxes),
        "spacing_xyz": list(spacing_xyz),
        "best_num_levels": best["num_levels"],
        "best_anchors_mm": best["anchors_mm"],
        "best_anchors_vox": best["anchors_vox"],
        "best_metrics": {
            "score": best["score"],
            "mean_best_iou": best["mean_best_iou"],
            "median_best_iou": best["median_best_iou"],
            "recall_iou_ge_0.25": best["recall_iou_ge_0.25"],
            "recall_iou_ge_0.50": best["recall_iou_ge_0.50"],
        },
        "all_results": results,
    }
    return summary


def save_anchor_search_result(result: Dict, out_json: Pathlike) -> None:
    out_json = Path(out_json)

    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj

    with open(out_json, "w") as f:
        json.dump(convert(result), f, indent=2)


if __name__ == "__main__":
    labels_tr = "/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted/labelsTr"

    result = find_best_anchors(
        labels_tr=labels_tr,
        spacing_xyz=(0.8, 0.8, 2.0),
        min_levels=3,
        max_levels=4,
        anchor_stat="q75",
        random_state=0,
        k_penalty=0.01,
    )

    print("Best number of levels:", result["best_num_levels"])
    print("Best anchors in voxels:")
    print(result["best_anchors_vox"])
    print("Best anchors in mm:")
    print(result["best_anchors_mm"])
    print("Best metrics:")
    print(result["best_metrics"])

    save_anchor_search_result(result, "/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted/anchor_search.json")
