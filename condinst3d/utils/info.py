from typing import Iterable, Union, Dict, Any
import torch
from monai.data import MetaTensor
from torch import Tensor
import os
import numpy as np
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


def extract_input_metadata(inputs: MetaTensor | Tensor) -> Dict[str, Any]:
    input_shape = tuple(inputs.shape)
    device = inputs.device

    input_meta = inputs.meta if isinstance(inputs, MetaTensor) else {}

    spacing = None
    pixdim = input_meta.get("pixdim", None)
    if pixdim is not None:
        spacing = tuple(pixdim[1:4])

    offsets = None
    if "location" in input_meta:
        offsets = input_meta["location"]

        if isinstance(offsets, (list, tuple, np.ndarray)):
            offsets = torch.as_tensor(offsets, device=device)
        elif not torch.is_tensor(offsets):
            offsets = torch.as_tensor(offsets, device=device)

        if offsets.ndim == 2 and offsets.shape[0] == 3:
            offsets = offsets.T

        offsets = offsets.to(device=device)

    return {
        "input_meta": input_meta,
        "input_shape": input_shape,
        "device": device,
        "offsets": offsets,
        "spacing": spacing,
    }
