from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Union
from dataclasses import dataclass
import json

import numpy as np
import SimpleITK as sitk
from tqdm import tqdm
from sklearn.cluster import KMeans


Pathlike = Union[str, Path]


@dataclass
class LesionSize:
    case_id: str
    instance_id: int
    voxel_count: int


def read_label(label_path: Path) -> sitk.Image:
    return sitk.ReadImage(str(label_path))


def extract_instance_sizes(label_path: Path) -> List[LesionSize]:
    """
    Extract voxel count for each nonzero instance ID in one label map.
    """
    img = read_label(label_path)
    arr = sitk.GetArrayFromImage(img)  # [z, y, x]

    case_id = label_path.name[:-7]
    instance_ids = np.unique(arr)
    instance_ids = instance_ids[instance_ids > 0]

    lesions: List[LesionSize] = []

    for inst_id in instance_ids:
        voxel_count = int((arr == inst_id).sum())
        if voxel_count <= 0:
            continue

        lesions.append(
            LesionSize(
                case_id=case_id,
                instance_id=int(inst_id),
                voxel_count=voxel_count,
            )
        )

    return lesions


def gather_all_instance_sizes(labels_tr: Path) -> List[LesionSize]:
    all_sizes: List[LesionSize] = []
    for label_path in tqdm(sorted(labels_tr.glob("*.nii.gz")), desc="Gathering lesion sizes"):
        all_sizes.extend(extract_instance_sizes(label_path))
    return all_sizes


def compute_size_bins_from_train(
    labels_tr: Pathlike,
    random_state: int = 0,
) -> Dict:
    """
    Learn 3 size bins (small/medium/large) from train instances using KMeans on log voxel count.
    Returns thresholds in voxel counts.
    """
    labels_tr = Path(labels_tr)
    lesions = gather_all_instance_sizes(labels_tr)

    if len(lesions) == 0:
        raise ValueError(f"No lesion instances found in {labels_tr}")

    voxel_counts = np.array([x.voxel_count for x in lesions], dtype=np.int64)

    # Cluster in log-space for more stable grouping across a long-tailed size distribution
    x = np.log(voxel_counts).reshape(-1, 1)

    km = KMeans(n_clusters=3, n_init=50, random_state=random_state)
    cluster_ids = km.fit_predict(x)

    # Order clusters by lesion size
    cluster_centers = km.cluster_centers_.reshape(-1)
    order = np.argsort(cluster_centers)

    # remap cluster id -> ordered id {0: small, 1: medium, 2: large}
    old_to_new = {old: new for new, old in enumerate(order)}
    ordered_cluster_ids = np.array([old_to_new[c] for c in cluster_ids], dtype=np.int64)

    small_sizes = voxel_counts[ordered_cluster_ids == 0]
    medium_sizes = voxel_counts[ordered_cluster_ids == 1]
    large_sizes = voxel_counts[ordered_cluster_ids == 2]

    if len(small_sizes) == 0 or len(medium_sizes) == 0 or len(large_sizes) == 0:
        raise RuntimeError("At least one learned cluster is empty. Check the training data.")

    # Define hard thresholds between adjacent ordered clusters
    # Use midpoint between max of lower cluster and min of upper cluster
    small_medium_thr = int(np.floor((small_sizes.max() + medium_sizes.min()) / 2.0))
    medium_large_thr = int(np.floor((medium_sizes.max() + large_sizes.min()) / 2.0))

    result = {
        "num_instances": int(len(lesions)),
        "cluster_centers_log_vox": cluster_centers[order].tolist(),
        "thresholds_vox": {
            "small_max": int(small_medium_thr),
            "medium_max": int(medium_large_thr),
        },
        "bins": {
            "small": [0, int(small_medium_thr)],
            "medium": [int(small_medium_thr) + 1, int(medium_large_thr)],
            "large": [int(medium_large_thr) + 1, int(voxel_counts.max())],
        },
        "summary": {
            "small": {
                "n": int(len(small_sizes)),
                "min": int(small_sizes.min()),
                "median": float(np.median(small_sizes)),
                "mean": float(np.mean(small_sizes)),
                "max": int(small_sizes.max()),
            },
            "medium": {
                "n": int(len(medium_sizes)),
                "min": int(medium_sizes.min()),
                "median": float(np.median(medium_sizes)),
                "mean": float(np.mean(medium_sizes)),
                "max": int(medium_sizes.max()),
            },
            "large": {
                "n": int(len(large_sizes)),
                "min": int(large_sizes.min()),
                "median": float(np.median(large_sizes)),
                "mean": float(np.mean(large_sizes)),
                "max": int(large_sizes.max()),
            },
            "all": {
                "min": int(voxel_counts.min()),
                "median": float(np.median(voxel_counts)),
                "mean": float(np.mean(voxel_counts)),
                "max": int(voxel_counts.max()),
            },
        },
        "per_instance": [
            {
                "case_id": lesions[i].case_id,
                "instance_id": lesions[i].instance_id,
                "voxel_count": int(lesions[i].voxel_count),
                "size_bin": ["small", "medium", "large"][int(ordered_cluster_ids[i])],
            }
            for i in range(len(lesions))
        ],
    }

    return result


def assign_size_bin(voxel_count: int, thresholds: Dict[str, int]) -> str:
    """
    Utility for later evaluation code.
    """
    if voxel_count <= thresholds["small_max"]:
        return "small"
    elif voxel_count <= thresholds["medium_max"]:
        return "medium"
    return "large"


def save_json(obj: Dict, out_json: Pathlike) -> None:
    out_json = Path(out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    with open(out_json, "w") as f:
        json.dump(obj, f, indent=2)


if __name__ == "__main__":
    labels_tr = "/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted/labelsTr"
    out_json = "/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted/lesion_size_bins_vox.json"

    result = compute_size_bins_from_train(
        labels_tr=labels_tr,
        random_state=0,
    )

    print("Number of instances:", result["num_instances"])
    print("Thresholds in voxels:")
    print(result["thresholds_vox"])
    print("Bins:")
    print(result["bins"])
    print("Summary:")
    print(json.dumps(result["summary"], indent=2))

    save_json(result, out_json)
    print(f"Saved to: {out_json}")
