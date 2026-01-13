import shutil
from loguru import logger
from sklearn.model_selection import train_test_split

from condinst3d.io.paths import Pathlike, Path, get_case_ids_from_dir


def create_test_split(splitted_dir: Pathlike,
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