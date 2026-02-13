from dataclasses import dataclass
from typing import Any, Dict, Hashable, Mapping, Optional

import numpy as np
import torch
from monai.transforms import MapTransform
from monai.config import KeysCollection
from condinst3d.evaluator.iou import box_intersection_over_union


@dataclass
class FilterAndUnpackPredsd(MapTransform):
    """
    Filter a prediction dict by score_threshold, unpack boxes/scores/labels into separate keys,
    and PERMUTE boxes to the desired format.

    Input pred dict fields:
      - pred_boxes:  (N, 6) in *source* order: (z1, y1, z2, y2, x1, x2)
      - pred_scores: (N,)
      - pred_labels: (N,)

    Output:
      - boxes:   (M, 6) in desired order: (x1, y1, z1, x2, y2, z2)
      - scores:  (M,)
      - classes: (M,)

    Notes:
      - Preserves numpy vs torch output: if any input arrays are torch, outputs are torch on same device.
      - If keep_topk is set, keeps top-k by score after thresholding.
      - If no predictions survive, outputs are empty arrays with correct shapes.
    """

    keys: KeysCollection  # keys that contain the pred dict(s)

    pred_boxes_field: str = "pred_boxes"
    pred_scores_field: str = "pred_scores"
    pred_labels_field: str = "pred_labels"

    score_threshold: float = 0.0
    keep_topk: Optional[int] = None

    out_boxes_key: str = "boxes"
    out_scores_key: str = "scores"
    out_labels_key: str = "classes"

    allow_missing_keys: bool = False
    strict: bool = True  # if False, fallback to empty outputs on errors

    # Source -> desired permutation:
    # src: (z1, y1, z2, y2, x1, x2)
    # dst: (x1, y1, z1, x2, y2, z2)
    _perm_src_to_dst = (4, 1, 0, 5, 3, 2)

    def __post_init__(self):
        super().__init__(self.keys, allow_missing_keys=self.allow_missing_keys)

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def _empty_outputs(self) -> Dict[str, np.ndarray]:
        return {
            self.out_boxes_key: np.zeros((0, 6), dtype=np.float32),
            self.out_scores_key: np.zeros((0,), dtype=np.float32),
            self.out_labels_key: np.zeros((0,), dtype=np.int64),
        }

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d: Dict[Hashable, Any] = dict(data)

        for key in self.key_iterator(d):
            try:
                pred = d.get(key, None)
                if pred is None:
                    outs = self._empty_outputs()
                    d[self.out_boxes_key] = outs[self.out_boxes_key]
                    d[self.out_scores_key] = outs[self.out_scores_key]
                    d[self.out_labels_key] = outs[self.out_labels_key]
                    continue

                if not isinstance(pred, dict):
                    raise TypeError(f"Expected a dict at key '{key}', got {type(pred)}")

                missing = [f for f in (self.pred_boxes_field, self.pred_scores_field, self.pred_labels_field) if f not in pred]
                if missing:
                    raise KeyError(f"Prediction dict at '{key}' missing fields: {missing}")

                boxes_raw = pred[self.pred_boxes_field]
                scores_raw = pred[self.pred_scores_field]
                labels_raw = pred[self.pred_labels_field]

                # Output torch if any input is torch
                out_as_torch = isinstance(boxes_raw, torch.Tensor) or isinstance(scores_raw, torch.Tensor) or isinstance(labels_raw, torch.Tensor)
                device = None
                if isinstance(boxes_raw, torch.Tensor):
                    device = boxes_raw.device
                elif isinstance(scores_raw, torch.Tensor):
                    device = scores_raw.device
                elif isinstance(labels_raw, torch.Tensor):
                    device = labels_raw.device

                boxes = self._to_numpy(boxes_raw).astype(np.float32, copy=False)
                scores = self._to_numpy(scores_raw).astype(np.float32, copy=False)
                labels = self._to_numpy(labels_raw).astype(np.int64, copy=False)

                if boxes.ndim != 2 or boxes.shape[1] != 6:
                    raise ValueError(f"'{self.pred_boxes_field}' must be (N,6), got {boxes.shape}")
                if scores.ndim != 1 or labels.ndim != 1:
                    raise ValueError(
                        f"'{self.pred_scores_field}' and '{self.pred_labels_field}' must be (N,), got {scores.shape} and {labels.shape}"
                    )
                if not (len(boxes) == len(scores) == len(labels)):
                    raise ValueError(f"Mismatched lengths: boxes={len(boxes)}, scores={len(scores)}, labels={len(labels)}")

                # Filter
                keep = scores >= float(self.score_threshold)
                boxes_f = boxes[keep]
                scores_f = scores[keep]
                labels_f = labels[keep]

                # Top-k
                if self.keep_topk is not None and boxes_f.shape[0] > int(self.keep_topk):
                    order = np.argsort(scores_f)[::-1][: int(self.keep_topk)]
                    boxes_f = boxes_f[order]
                    scores_f = scores_f[order]
                    labels_f = labels_f[order]

                # PERMUTE boxes: (z1,y1,z2,y2,x1,x2) -> (x1,y1,z1,x2,y2,z2)
                if boxes_f.shape[0] > 0:
                    boxes_f = boxes_f[:, self._perm_src_to_dst]

                # Optional sanity check: x1<=x2, y1<=y2, z1<=z2
                if boxes_f.shape[0] > 0:
                    if not (np.all(boxes_f[:, 0] <= boxes_f[:, 3]) and
                            np.all(boxes_f[:, 1] <= boxes_f[:, 4]) and
                            np.all(boxes_f[:, 2] <= boxes_f[:, 5])):
                        raise ValueError(
                            "After permutation, box ordering check failed (x1<=x2,y1<=y2,z1<=z2). "
                            "This suggests the source box format is not (z1,y1,z2,y2,x1,x2)."
                        )

                # Write outputs
                if out_as_torch:
                    d[self.out_boxes_key] = torch.as_tensor(boxes_f, device=device, dtype=torch.float32)
                    d[self.out_scores_key] = torch.as_tensor(scores_f, device=device, dtype=torch.float32)
                    d[self.out_labels_key] = torch.as_tensor(labels_f, device=device, dtype=torch.long)
                else:
                    d[self.out_boxes_key] = boxes_f
                    d[self.out_scores_key] = scores_f
                    d[self.out_labels_key] = labels_f

            except Exception:
                if self.strict:
                    raise
                outs = self._empty_outputs()
                d[self.out_boxes_key] = outs[self.out_boxes_key]
                d[self.out_scores_key] = outs[self.out_scores_key]
                d[self.out_labels_key] = outs[self.out_labels_key]

        return d


def _iou_3d_xyzxyz(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """
    a: (N,6) xyzxyz
    b: (M,6) xyzxyz
    returns iou: (N,M)
    """
    a = a.astype(np.float32, copy=False)
    b = b.astype(np.float32, copy=False)

    # expand for broadcast
    ax1, ay1, az1, ax2, ay2, az2 = [a[:, i][:, None] for i in range(6)]
    bx1, by1, bz1, bx2, by2, bz2 = [b[:, i][None, :] for i in range(6)]

    ix1 = np.maximum(ax1, bx1)
    iy1 = np.maximum(ay1, by1)
    iz1 = np.maximum(az1, bz1)
    ix2 = np.minimum(ax2, bx2)
    iy2 = np.minimum(ay2, by2)
    iz2 = np.minimum(az2, bz2)

    iw = np.maximum(ix2 - ix1, 0.0)
    ih = np.maximum(iy2 - iy1, 0.0)
    id = np.maximum(iz2 - iz1, 0.0)

    inter = iw * ih * id

    a_vol = np.maximum(ax2 - ax1, 0.0) * np.maximum(ay2 - ay1, 0.0) * np.maximum(az2 - az1, 0.0)
    b_vol = np.maximum(bx2 - bx1, 0.0) * np.maximum(by2 - by1, 0.0) * np.maximum(bz2 - bz1, 0.0)

    union = a_vol + b_vol - inter
    return inter / (union + eps)


@dataclass
class MatchSegBoxesToPredScoresd(MapTransform):
    """
    Match segmentation-derived boxes to predicted boxes (with scores) using IoU.

    Inputs:
      - seg_boxes_key:  (Ns,6) boxes from mask (xyzxyz)
      - pred_boxes_key: (Np,6) predicted boxes (xyzxyz)
      - pred_scores_key:(Np,)  predicted scores

    Output:
      - out_scores_key: (Ns,) per-seg-box score:
          for each seg box i, find the pred box j with max IoU;
          if max IoU >= iou_threshold -> score = pred_scores[j]
          else -> default_score.

    Options:
      - choose="max_iou" (default): pick score of the pred box with highest IoU
      - If you want "max_score among IoU>=th" instead, tell me and I’ll swap strategy.

    Notes:
      - Works with numpy or torch.
      - Returns torch if any input is torch (on the same device).
      - Empty inputs are handled (returns empty scores).
    """

    seg_boxes_key: Hashable
    pred_boxes_key: Hashable
    pred_scores_key: Hashable

    out_scores_key: Hashable = "seg_box_scores"

    iou_threshold: float = 0.1
    default_score: float = 0.0

    allow_missing_keys: bool = False
    strict: bool = True

    def __post_init__(self):
        super().__init__([self.seg_boxes_key, self.pred_boxes_key, self.pred_scores_key],
                         allow_missing_keys=self.allow_missing_keys)

    @staticmethod
    def _to_numpy(x: Any) -> np.ndarray:
        if x is None:
            return None
        if isinstance(x, np.ndarray):
            return x
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
        return np.asarray(x)

    def __call__(self, data: Mapping[Hashable, Any]) -> Dict[Hashable, Any]:
        d = dict(data)
        try:
            seg_b = d.get(self.seg_boxes_key, None)
            pred_b = d.get(self.pred_boxes_key, None)
            pred_s = d.get(self.pred_scores_key, None)

            if seg_b is None:
                d[self.out_scores_key] = None
                return d

            out_as_torch = isinstance(seg_b, torch.Tensor) or isinstance(pred_b, torch.Tensor) or isinstance(pred_s, torch.Tensor)
            device = seg_b.device if isinstance(seg_b, torch.Tensor) else (
                pred_b.device if isinstance(pred_b, torch.Tensor) else (
                    pred_s.device if isinstance(pred_s, torch.Tensor) else None
                )
            )

            seg_boxes = self._to_numpy(seg_b).astype(np.float32, copy=False)
            if seg_boxes.ndim != 2 or seg_boxes.shape[1] != 6:
                raise ValueError(f"seg boxes must be (N,6) xyzxyz, got {seg_boxes.shape}")

            Ns = int(seg_boxes.shape[0])
            if Ns == 0:
                scores_out = np.zeros((0,), dtype=np.float32)
                d[self.out_scores_key] = torch.as_tensor(scores_out, device=device) if out_as_torch else scores_out
                return d

            # If no preds, all defaults
            if pred_b is None or pred_s is None:
                scores_out = np.full((Ns,), float(self.default_score), dtype=np.float32)
                d[self.out_scores_key] = torch.as_tensor(scores_out, device=device) if out_as_torch else scores_out
                return d

            pred_boxes = self._to_numpy(pred_b).astype(np.float32, copy=False)
            pred_scores = self._to_numpy(pred_s).astype(np.float32, copy=False)

            if pred_boxes.ndim != 2 or pred_boxes.shape[1] != 6:
                raise ValueError(f"pred boxes must be (N,6) xyzxyz, got {pred_boxes.shape}")
            if pred_scores.ndim != 1:
                raise ValueError(f"pred scores must be (N,), got {pred_scores.shape}")
            if pred_boxes.shape[0] != pred_scores.shape[0]:
                raise ValueError(f"pred boxes and scores length mismatch: {pred_boxes.shape[0]} vs {pred_scores.shape[0]}")

            Np = int(pred_boxes.shape[0])
            if Np == 0:
                scores_out = np.full((Ns,), float(self.default_score), dtype=np.float32)
                d[self.out_scores_key] = torch.as_tensor(scores_out, device=device) if out_as_torch else scores_out
                return d

            # IoU matrix (Ns, Np)
            ious = _iou_3d_xyzxyz(seg_boxes, pred_boxes)

            # For each seg box, take best match
            best_j = np.argmax(ious, axis=1)                 # (Ns,)
            best_iou = ious[np.arange(Ns), best_j]           # (Ns,)

            scores_out = np.full((Ns,), float(self.default_score), dtype=np.float32)
            good = best_iou >= float(self.iou_threshold)
            scores_out[good] = pred_scores[best_j[good]]

            if out_as_torch:
                d[self.out_scores_key] = torch.as_tensor(scores_out, device=device, dtype=torch.float32)
            else:
                d[self.out_scores_key] = scores_out

            # (Optional) you might also want the matched indices / IoUs:
            # d[self.out_scores_key + "_best_iou"] = best_iou
            # d[self.out_scores_key + "_best_j"] = best_j

            return d

        except Exception:
            if self.strict:
                raise
            d[self.out_scores_key] = None
            return d
