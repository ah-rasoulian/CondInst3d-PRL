import torch
from torch import Tensor

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
