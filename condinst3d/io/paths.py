from typing import Union, List
from pathlib import Path
import os

Pathlike = Union[Path, str]

def get_case_ids_from_dir(dir_path: Path, unique: bool = True,
                          remove_modality: bool = True, join: bool = False,
                          pattern="*.nii.gz") -> List[str]:
    """
    Get all case ids from a single folder

    Args:
        dir_path: path to folder
        unique: remove all duplicates
        remove_modality: remove the modality string from the filename
        join: append case ids to directory path
        pattern: regular expression used to select files

    Returns:
        List[str]: all case ids inside the folder
    """
    files = map(str, list(Path(dir_path).glob(pattern)))
    case_ids = [get_case_id_from_path(f, remove_modality=remove_modality) for f in files]
    if unique:
        case_ids = list(set(case_ids))
    if join:
        case_ids = [os.path.join(dir_path, c) for c in case_ids]
    return case_ids

def get_case_id_from_path(file_path: Pathlike, remove_modality: bool = True) -> str:
    """
    Get case of from path to file

    Args:
        file_path (str): path to file as string
        remove_modality (bool): remove the modality string from the filename
            (only used if file ends with .nii.gz)

    Returns:
        str: case id
    """
    file_name = str(file_path).rsplit(os.path.sep, 1)[1]
    return get_case_id_from_file(file_name, remove_modality=remove_modality)

def get_case_id_from_file(file_name: str, remove_modality: bool = True) -> str:
    """
    Cut of ".nii.gz" from file name

    Args:
        file_name (str): name of file with .nii.gz ending
        remove_modality (bool): remove the modality string from the filename

    Returns:
        str: name of file without ending
    """
    if file_name.endswith(".nii.gz"):
        file_name = file_name.rsplit(".", 2)[0]
    else:
        file_name = file_name.rsplit(".", 1)[0]

    if remove_modality:
        file_name = file_name[:-5]
    return file_name

def is_existing_path(path):
    return isinstance(path, (str, os.PathLike)) and os.path.exists(path)

def find_minctools_path(binary="mincresample"):
    binaries = [
        "/app/miniconda/envs/minc/bin/"
        "/opt/minc-toolkit/0.3.16/bin/",
        "/opt/minc/1.9.15/bin/",
        "/usr/bin/",
        "/scratch/03/ahrasoulian/other/miniconda3/envs/minc/bin/",
    ]
    mincresample_binary = None
    for b in binaries:
        if is_existing_path(b + binary):
            mincresample_binary = b
            break
    if mincresample_binary is None:
        raise ValueError("could not find mincresample on the system")
    return mincresample_binary + binary
