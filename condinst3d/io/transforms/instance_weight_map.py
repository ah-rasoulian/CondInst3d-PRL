from __future__ import annotations
from typing import Hashable
import torch
from monai.transforms import MapTransform


class MakeBalancedInstanceWeightMapd(MapTransform):
    """
    Creates a sampling weight map from an instance-id mask.

    Expected spatial order: (X, Y, Z).
    Input instance mask shape: [X,Y,Z] or [1,X,Y,Z].

    If instances exist:
      - balanced per-instance weights (NO center bias):
          voxels of instance k get 1/size(k), background stays 0.

    If NO instances exist:
      - random-but-slightly-center-biased weights:
          w = alpha_uniform * uniform + (1 - alpha_uniform) * gaussian_center
        This guarantees borders keep probability mass as long as alpha_uniform > 0.

    Args:
        instance_key: key for instance id map.
        out_key: key to store weight map (float32), shape [X,Y,Z].
        center_sigma_scale: sigma = min(X,Y,Z) / center_sigma_scale (bigger -> flatter).
        alpha_uniform: mixture weight for uniform component in negative cases (0..1).
        eps: numerical stability.
    """

    def __init__(
        self,
        instance_key: Hashable = "instance_mask",
        out_key: Hashable = "inst_wmap",
        *,
        center_sigma_scale: float = 3.0,
        alpha_uniform: float = 0.25,
        eps: float = 1e-8,
        allow_missing_keys: bool = False,
    ):
        super().__init__([instance_key], allow_missing_keys)
        self.instance_key = instance_key
        self.out_key = out_key
        self.center_sigma_scale = float(center_sigma_scale)
        self.alpha_uniform = float(alpha_uniform)
        self.eps = float(eps)

        if not (0.0 <= self.alpha_uniform <= 1.0):
            raise ValueError(f"alpha_uniform must be in [0,1], got {self.alpha_uniform}")

    def __call__(self, data):
        d = dict(data)
        inst = d[self.instance_key]
        if not torch.is_tensor(inst):
            inst = torch.as_tensor(inst)

        # squeeze [1,X,Y,Z] -> [X,Y,Z]
        if inst.ndim == 4 and inst.shape[0] == 1:
            inst = inst.squeeze(0)

        if inst.ndim != 3:
            raise ValueError(
                f"Expected instance mask [X,Y,Z] (or [1,X,Y,Z]), got shape={tuple(inst.shape)}"
            )

        X, Y, Z = map(int, inst.shape)
        device = inst.device

        w = torch.zeros((X, Y, Z), device=device, dtype=torch.float32)

        ids = torch.unique(inst)
        ids = ids[ids > 0]  # skip background

        if ids.numel() == 0:
            # Negative case: mixture of uniform + center Gaussian (keeps borders alive).
            uniform = torch.ones_like(w)

            x = torch.arange(X, device=device, dtype=torch.float32)
            y = torch.arange(Y, device=device, dtype=torch.float32)
            z = torch.arange(Z, device=device, dtype=torch.float32)

            x0, y0, z0 = (X - 1) / 2.0, (Y - 1) / 2.0, (Z - 1) / 2.0
            Xg, Yg, Zg = torch.meshgrid(x, y, z, indexing="ij")

            r2 = (Xg - x0) ** 2 + (Yg - y0) ** 2 + (Zg - z0) ** 2
            sigma = max(min(X, Y, Z) / self.center_sigma_scale, 1.0)
            center = torch.exp(-0.5 * r2 / (sigma ** 2))  # (0,1]

            w = self.alpha_uniform * uniform + (1.0 - self.alpha_uniform) * center

            # safety: if something went wrong, fall back to uniform
            if float(w.sum().item()) < self.eps:
                w = uniform

        else:
            # Positive case: balanced per-instance weights, background stays 0.
            for k in ids:
                m = inst == k
                n = int(m.sum())
                if n > 0:
                    w[m] = 1.0 / float(n)

            # safety fallback
            if float(w.sum().item()) < self.eps:
                w = (inst > 0).to(torch.float32)

        d[self.out_key] = w.unsqueeze(0)
        return d
