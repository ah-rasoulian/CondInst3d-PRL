from typing import Iterable, Union
import torch
import os
from tqdm import tqdm

DeviceType = Union[str, torch.device]
COMPUTE_DTYPE = torch.float32
INF = float("inf")

def maybe_verbose_iterable(data: Iterable, **kwargs) -> Iterable:
    """
    If verbose flag of nndet is enabled, uses tqdm to create a
    progress bar

    Args:
        data: iterable to wrap
        **kwargs: keyword arguments passed to tqdm

    Returns:
        Iterable: maybe iterable with progress bar atteched to it
    """
    if bool(int(os.getenv("det_verbose", 1))):
        return tqdm(data, **kwargs)
    else:
        return data