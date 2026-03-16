from __future__ import annotations

from typing import Optional, Tuple
import torch
from torch import Tensor
from skimage.filters import threshold_otsu
from monai.data import MetaTensor
import numpy as np
from scipy import ndimage

COMPUTE_DTYPE = torch.float32


def relabel_sequential(mask: Tensor, exclude_background: bool = True) -> Tensor:
    """
    Relabels unique labels to 1..K (keeps 0 as background if exclude_background=True).
    Works for any shape (e.g. [H,W,D] or [C,H,W,D]).
    """
    uniq = torch.unique(mask)
    if exclude_background:
        uniq_fg = uniq[uniq != 0]
        if uniq_fg.numel() == 0:
            return mask.clone()
        uniq_sorted, _ = torch.sort(uniq_fg)
        # map label -> rank+1 using searchsorted
        flat = mask.reshape(-1)
        out = torch.zeros_like(flat)
        fg = flat != 0
        out[fg] = torch.searchsorted(uniq_sorted, flat[fg]).to(out.dtype) + 1
        return out.view_as(mask)
    else:
        uniq_sorted, _ = torch.sort(uniq)
        flat = mask.reshape(-1)
        out = torch.searchsorted(uniq_sorted, flat).to(mask.dtype) + 1
        return out.view_as(mask)


def build_gt_cluster_ids(
    semantic_mask: Tensor,
    instance_mask: Tensor,
    *,
    connectivity: int = 26,
    min_instances_in_cluster: int = 2,
    instance_ids: Optional[Tensor] = None,
    background_label: int = 0,
) -> Tensor:
    """
    Build gt_cluster_ids for confluent clusters.

    A "cluster" is defined as a connected component (CC) of the *semantic* mask.
    For each CC, we look at how many distinct GT instances (labels in instance_mask)
    are present inside that CC. If that count >= min_instances_in_cluster, then all
    those instance labels are marked as belonging to a confluent cluster.

    Returns:
      gt_cluster_ids: LongTensor [N_gt], where:
          -1  => GT instance is not in a confluent cluster
          0.. => cluster id (contiguous) for confluent clusters

    Assumptions:
      - instance_mask is an integer-labeled mask with background_label for background
        and positive labels for each GT instance.
      - If `instance_ids` is None, we assume GT instance labels are 1..N_gt (dense),
        and return ids for that order: label=1 maps to index 0, etc.
      - If your GT labels are not dense or not 1..N_gt, pass `instance_ids`, a 1D tensor
        listing the GT instance labels in the same order as your IoU matrix columns.

    Notes:
      - Works for 2D or 3D (or ND), but connectivity is handled for 2D/3D:
          connectivity=4/8 for 2D, 6/18/26 for 3D.
      - Uses scipy.ndimage for connected components. If scipy isn't available, it raises.
    """
    if semantic_mask.shape != instance_mask.shape:
        raise ValueError(
            f"semantic_mask and instance_mask must have the same shape, got "
            f"{tuple(semantic_mask.shape)} vs {tuple(instance_mask.shape)}"
        )

    if semantic_mask.dtype != torch.bool:
        sem = semantic_mask != 0
    else:
        sem = semantic_mask

    if instance_mask.dtype not in (torch.int16, torch.int32, torch.int64, torch.uint8):
        # allow floats but force to int (common if loaded as float)
        inst = instance_mask.to(torch.int64)
    else:
        inst = instance_mask.to(torch.int64)

    # Determine which instance labels define the GT ordering
    if instance_ids is None:
        # Assume dense labels 1..N_gt
        max_id = int(inst.max().item()) if inst.numel() else 0
        if max_id <= 0:
            return torch.empty((0,), dtype=torch.long, device=inst.device)
        instance_ids = torch.arange(1, max_id + 1, device=inst.device, dtype=torch.long)
    else:
        if instance_ids.dim() != 1:
            raise ValueError(f"instance_ids must be 1D, got {tuple(instance_ids.shape)}")
        instance_ids = instance_ids.to(device=inst.device, dtype=torch.long)

    N_gt = int(instance_ids.numel())
    gt_cluster_ids = torch.full((N_gt,), -1, device=inst.device, dtype=torch.long)

    # If nothing foreground, return all -1
    if not torch.any(sem):
        return gt_cluster_ids

    # --- connected components on semantic mask (CPU via scipy) ---
    try:
        import numpy as np
        import scipy.ndimage as ndi
    except Exception as e:
        raise RuntimeError(
            "scipy is required for connected-components labeling in build_gt_cluster_ids(). "
            "Install scipy or tell me your stack (MONAI/cc3d/skimage) and I'll adapt."
        ) from e

    sem_np = sem.detach().to("cpu").numpy().astype(np.bool_)
    inst_np = inst.detach().to("cpu").numpy().astype(np.int64)

    ndim = sem_np.ndim
    if ndim not in (2, 3):
        # For ND, we approximate with full connectivity (all ones) if requested.
        # You can adjust if you truly use ND > 3.
        structure = np.ones((3,) * ndim, dtype=np.int8)
    else:
        if ndim == 2:
            if connectivity == 4:
                structure = np.array([[0, 1, 0],
                                      [1, 1, 1],
                                      [0, 1, 0]], dtype=np.int8)
            elif connectivity == 8:
                structure = np.ones((3, 3), dtype=np.int8)
            else:
                raise ValueError("For 2D, connectivity must be 4 or 8.")
        else:  # ndim == 3
            if connectivity == 6:
                structure = np.zeros((3, 3, 3), dtype=np.int8)
                structure[1, 1, 1] = 1
                structure[0, 1, 1] = 1
                structure[2, 1, 1] = 1
                structure[1, 0, 1] = 1
                structure[1, 2, 1] = 1
                structure[1, 1, 0] = 1
                structure[1, 1, 2] = 1
            elif connectivity == 18:
                structure = np.ones((3, 3, 3), dtype=np.int8)
                # remove the 8 corners to get 18-connectivity
                corners = [(0, 0, 0), (0, 0, 2), (0, 2, 0), (0, 2, 2),
                           (2, 0, 0), (2, 0, 2), (2, 2, 0), (2, 2, 2)]
                for c in corners:
                    structure[c] = 0
            elif connectivity == 26:
                structure = np.ones((3, 3, 3), dtype=np.int8)
            else:
                raise ValueError("For 3D, connectivity must be 6, 18, or 26.")

    cc_labels, num_cc = ndi.label(sem_np, structure=structure)

    # Build a fast map label -> gt index (based on instance_ids ordering)
    # instance_ids are labels in inst_np.
    id_to_index = {int(lab): idx for idx, lab in enumerate(instance_ids.detach().to("cpu").tolist())}

    cluster_id = 0
    for cc in range(1, num_cc + 1):
        mask_cc = (cc_labels == cc)
        if not mask_cc.any():
            continue

        # instance labels inside this connected component (excluding background)
        labs = np.unique(inst_np[mask_cc])
        labs = labs[labs != background_label]
        if labs.size < min_instances_in_cluster:
            continue

        # Only keep labels that exist in instance_ids (safety)
        member_indices = []
        for lab in labs.tolist():
            idx = id_to_index.get(int(lab), None)
            if idx is not None:
                member_indices.append(idx)

        if len(member_indices) < min_instances_in_cluster:
            continue

        gt_cluster_ids[torch.tensor(member_indices, device=gt_cluster_ids.device, dtype=torch.long)] = cluster_id
        cluster_id += 1

    return gt_cluster_ids


def connected_components(
        binary_mask: Tensor,
        connectivity: int = 26,
) -> Tensor:
    """
    Connected components for a single 3D binary mask.

    Parameters
    ----------
    binary_mask:
        Bool or binary tensor of shape [H, W, D].

    connectivity:
        One of {6, 18, 26} for 3D connectivity.

    Returns
    -------
    Tensor [H, W, D] of dtype long, where:
      - 0 is background
      - 1..K are connected component ids
    """
    if binary_mask.ndim != 3:
        raise ValueError(
            f"connected_components expects a 3D tensor [H,W,D], got shape {tuple(binary_mask.shape)}"
        )

    binary_mask_np = binary_mask.detach().to(dtype=torch.bool, device="cpu").numpy()

    if connectivity == 6:
        structure = ndimage.generate_binary_structure(rank=3, connectivity=1)
    elif connectivity == 18:
        # scipy does not expose “18” directly as a named mode; this structure works for it
        structure = ndimage.generate_binary_structure(rank=3, connectivity=2)
        structure[0, 0, 0] = 0
        structure[0, 0, 2] = 0
        structure[0, 2, 0] = 0
        structure[0, 2, 2] = 0
        structure[2, 0, 0] = 0
        structure[2, 0, 2] = 0
        structure[2, 2, 0] = 0
        structure[2, 2, 2] = 0
    elif connectivity == 26:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
    else:
        raise ValueError(f"connectivity must be one of {{6,18,26}}, got {connectivity}")

    labeled_np, _ = ndimage.label(binary_mask_np, structure=structure)

    return torch.as_tensor(labeled_np, device=binary_mask.device, dtype=torch.long)