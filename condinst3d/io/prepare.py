import shutil
from loguru import logger
from sklearn.model_selection import train_test_split
from collections import OrderedDict
from condinst3d.io.paths import Pathlike, Path, get_case_ids_from_dir
import pickle

from pathlib import Path
import shutil
import logging
from typing import Union
from collections import Counter
from tqdm import tqdm
import numpy as np
import SimpleITK as sitk
from scipy import ndimage
from sklearn.model_selection import train_test_split, StratifiedKFold

logger = logging.getLogger(__name__)
Pathlike = Union[str, Path]


def _read_label(label_path: Path) -> np.ndarray:
    img = sitk.ReadImage(str(label_path))
    arr = sitk.GetArrayFromImage(img)
    return arr


def _case_instance_stats(label_path: Path) -> tuple[int, int]:
    """
    Returns:
        n_lesions: number of unique nonzero instance IDs
        n_confluent: number of binary connected components that contain
                     more than one instance ID
    """
    arr = _read_label(label_path)

    instance_ids = np.unique(arr)
    instance_ids = instance_ids[instance_ids > 0]
    n_lesions = int(len(instance_ids))

    if n_lesions == 0:
        return 0, 0

    binary = arr > 0

    # 3D full connectivity
    structure = ndimage.generate_binary_structure(rank=3, connectivity=3)
    cc, n_cc = ndimage.label(binary, structure=structure)

    n_confluent = 0
    for cc_id in range(1, n_cc + 1):
        insts_in_cc = np.unique(arr[cc == cc_id])
        insts_in_cc = insts_in_cc[insts_in_cc > 0]
        if len(insts_in_cc) > 1:
            n_confluent += 1

    return n_lesions, n_confluent


def _bin_lesion_count(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    if 3 <= n <= 5:
        return "3-5"
    if 6 <= n <= 9:
        return "6-9"
    return "10+"


def _bin_confluent_count(n: int) -> str:
    if n == 0:
        return "0"
    if n == 1:
        return "1"
    if n == 2:
        return "2"
    return "3+"


def _make_stratum(n_lesions: int, n_confluent: int) -> str:
    lesion_bin = _bin_lesion_count(n_lesions)
    confluent_bin = _bin_confluent_count(n_confluent)
    return f"L{lesion_bin}_C{confluent_bin}"


def _collapse_rare_strata(
    strata: list[str],
    test_size: float,
) -> list[str]:
    """
    Merge rare strata into broader buckets so sklearn stratify won't fail.

    A stratum is considered too rare if it cannot contribute at least one
    sample to both train and test.
    """
    counts = Counter(strata)

    min_needed = max(2, int(np.ceil(1 / test_size)))
    # Example:
    # test_size=0.3 -> ceil(3.33)=4
    # a class with only 2-3 examples often causes bad or unstable splits

    collapsed = []
    for s in strata:
        if counts[s] >= min_needed:
            collapsed.append(s)
        else:
            # fallback: preserve only confluent bin first
            # e.g. L3-5_C1 -> C1
            confluent_part = s.split("_")[1]
            collapsed.append(f"RARE_{confluent_part}")

    # second pass if still too rare
    counts2 = Counter(collapsed)
    final = []
    for s in collapsed:
        if counts2[s] >= 2:
            final.append(s)
        else:
            final.append("RARE")
    return final

def _collapse_rare_strata_for_kfold(strata: list[str], n_splits: int) -> list[str]:
    """
    For StratifiedKFold, every stratum should ideally have at least n_splits samples.
    Rare strata are progressively merged into coarser groups.
    """
    counts = Counter(strata)

    collapsed = []
    for s in strata:
        if counts[s] >= n_splits:
            collapsed.append(s)
        else:
            # keep only confluent part, e.g. L10+_C2 -> RARE_C2
            confluent_part = s.split("_")[1]
            collapsed.append(f"RARE_{confluent_part}")

    counts2 = Counter(collapsed)
    final = []
    for s in collapsed:
        if counts2[s] >= n_splits:
            final.append(s)
        else:
            final.append("RARE")
    return final


def create_splits_final_pkl(
    splitted_dir: Pathlike,
    out_path: Pathlike,
    n_splits: int = 5,
    random_state: int = 0,
    shuffle: bool = True,
):
    """
    Create nnU-Net-style splits_final.pkl with stratification based on:
      - number of lesions
      - number of confluent lesions
    """
    splitted_dir = Path(splitted_dir)
    images_tr = splitted_dir / "imagesTr"
    labels_tr = splitted_dir / "labelsTr"

    if not images_tr.is_dir():
        raise ValueError(f"No dir with training images found: {images_tr}")
    if not labels_tr.is_dir():
        raise ValueError(f"No dir with training labels found: {labels_tr}")

    case_ids = sorted(get_case_ids_from_dir(images_tr, remove_modality=True))
    logger.info(f"Found {len(case_ids)} training cases")

    case_stats = []
    for cid in tqdm(case_ids):
        label_path = labels_tr / f"{cid}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label for case {cid}: {label_path}")

        n_lesions, n_confluent = _case_instance_stats(label_path)
        stratum = _make_stratum(n_lesions, n_confluent)

        case_stats.append({
            "cid": cid,
            "n_lesions": n_lesions,
            "n_confluent": n_confluent,
            "stratum": stratum,
        })

    raw_strata = [x["stratum"] for x in case_stats]
    strata = _collapse_rare_strata_for_kfold(raw_strata, n_splits=n_splits)

    logger.info("Final fold stratification distribution:")
    for k, v in sorted(Counter(strata).items()):
        logger.info(f"  {k}: {v}")

    case_ids_arr = np.array([x["cid"] for x in case_stats])
    strata_arr = np.array(strata)

    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state,
    )

    splits = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(case_ids_arr, strata_arr)):
        train_cases = np.sort(case_ids_arr[train_idx])
        val_cases = np.sort(case_ids_arr[val_idx])

        split = OrderedDict()
        split["train"] = train_cases
        split["val"] = val_cases
        splits.append(split)

        logger.info(
            f"Fold {fold_idx}: train={len(train_cases)} val={len(val_cases)}"
        )

    with open(out_path, "wb") as f:
        pickle.dump(splits, f)

    logger.info(f"Saved splits to {out_path}")
    return splits


def create_stratified_test_split(
    splitted_dir: Pathlike,
    num_modalities: int,
    test_size: float = 0.3,
    random_state: int = 0,
    shuffle: bool = True,
):
    """
    Create a stratified artificial test split from instance masks.

    Stratification is based on:
      - number of lesions
      - number of confluent lesions

    A confluent lesion here means a binary connected component that contains
    more than one instance ID.
    """
    images_tr = Path(splitted_dir) / "imagesTr"
    labels_tr = Path(splitted_dir) / "labelsTr"
    images_ts = Path(splitted_dir) / "imagesTs"
    labels_ts = Path(splitted_dir) / "labelsTs"

    if not images_tr.is_dir():
        raise ValueError(f"No dir with training images found {images_tr}")
    if not labels_tr.is_dir():
        raise ValueError(f"No dir with training labels found {labels_tr}")

    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(get_case_ids_from_dir(images_tr, remove_modality=True))
    logger.info(f"Found {len(case_ids)} cases to split")

    # Compute per-case stats
    case_stats = []
    for cid in tqdm(case_ids):
        label_path = labels_tr / f"{cid}.nii.gz"
        if not label_path.is_file():
            raise FileNotFoundError(f"Missing label file for case {cid}: {label_path}")

        n_lesions, n_confluent = _case_instance_stats(label_path)
        stratum = _make_stratum(n_lesions, n_confluent)

        case_stats.append({
            "cid": cid,
            "n_lesions": n_lesions,
            "n_confluent": n_confluent,
            "stratum": stratum,
        })

    strata = [x["stratum"] for x in case_stats]
    strata = _collapse_rare_strata(strata, test_size=test_size)

    logger.info("Final stratification distribution:")
    for k, v in sorted(Counter(strata).items()):
        logger.info(f"  {k}: {v}")

    if shuffle:
        train_ids, test_ids = train_test_split(
            [x["cid"] for x in case_stats],
            test_size=test_size,
            random_state=random_state,
            shuffle=True,
            stratify=strata,
        )
    else:
        # true stratification requires shuffle in sklearn
        raise ValueError("shuffle=False is not compatible with meaningful stratified splitting.")

    logger.info(f"Using {len(train_ids)} cases for training and {len(test_ids)} for testing.")
    logger.info(f"Test IDs: {test_ids}")

    # Optional: report resulting stats
    stats_by_id = {x["cid"]: x for x in case_stats}
    train_lesions = [stats_by_id[c]["n_lesions"] for c in train_ids]
    test_lesions = [stats_by_id[c]["n_lesions"] for c in test_ids]
    train_confluent = [stats_by_id[c]["n_confluent"] for c in train_ids]
    test_confluent = [stats_by_id[c]["n_confluent"] for c in test_ids]

    logger.info(
        f"Train lesions mean={np.mean(train_lesions):.2f}, "
        f"test lesions mean={np.mean(test_lesions):.2f}"
    )
    logger.info(
        f"Train confluent mean={np.mean(train_confluent):.2f}, "
        f"test confluent mean={np.mean(test_confluent):.2f}"
    )

    # Move test files
    for cid in test_ids:
        for modality in range(num_modalities):
            shutil.move(
                images_tr / f"{cid}_{modality:04d}.nii.gz",
                images_ts / f"{cid}_{modality:04d}.nii.gz",
            )

        shutil.move(labels_tr / f"{cid}.nii.gz", labels_ts / f"{cid}.nii.gz")

        json_path = labels_tr / f"{cid}.json"
        if json_path.is_file():
            shutil.move(json_path, labels_ts / f"{cid}.json")


def create_random_test_split(splitted_dir: Pathlike,
                             num_modalities: int,
                             test_size: float = 0.3,
                             random_state: int = 0,
                             shuffle: bool = True,
                            ):
    """
    Helper function to create an artificial test split from the splitted data

    Args:
        splitted_dir: path to directory with splitted data. `imagesTr` and
            `labelsTr` need to exist beforehand. `imagesTs` and `labelsTs`
            will be created automatically.
        num_modalities: number of modalities
        test_size: size of test set, needs to be a value between 0 and 1
        seed: seed for splitting
        shuffle: shuffle data
    """
    images_tr = Path(splitted_dir) / "imagesTr"
    labels_tr = Path(splitted_dir) / "labelsTr"
    images_ts = Path(splitted_dir) / "imagesTs"
    labels_ts = Path(splitted_dir) / "labelsTs"

    if not images_tr.is_dir():
        raise ValueError(f"No dir with training images found {images_tr}")
    if not labels_tr.is_dir():
        raise ValueError(f"No dir with training labels found {labels_tr}")
    images_ts.mkdir(parents=True, exist_ok=True)
    labels_ts.mkdir(parents=True, exist_ok=True)

    case_ids = sorted(get_case_ids_from_dir(images_tr, remove_modality=True))
    logger.info(f"Found {len(case_ids)} to split")

    train_ids, test_ids = train_test_split(
        case_ids, test_size=test_size, random_state=random_state, shuffle=shuffle)
    logger.info(f"Using {train_ids} for training and {test_ids} for testing.")

    for cid in test_ids:
        for modality in range(num_modalities):
            shutil.move(images_tr / f"{cid}_{modality:04d}.nii.gz",
                        images_ts / f"{cid}_{modality:04d}.nii.gz")
        shutil.move(labels_tr / f"{cid}.nii.gz", labels_ts / f"{cid}.nii.gz")
        if (labels_tr / f"{cid}.json").is_file():
            shutil.move(labels_tr / f"{cid}.json", labels_ts / f"{cid}.json")