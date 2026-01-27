from typing import Literal, Callable, Sequence, Tuple
from monai.apps.detection.utils.ATSS_matcher import Matcher
from monai.data.box_utils import box_iou, centers_in_boxes, box_centers
import torch
from torch import Tensor
from monai.utils.type_conversion import convert_data_type, convert_to_dst_type, convert_to_tensor
from condinst3d.utils.info import  DeviceType, COMPUTE_DTYPE, INF
from condinst3d.utils.spatial import centers_in_boxes, boxes_center_distance
import logging


def compute_locations(
    width: int,
    height: int,
    depth: int,
    stride: Tuple[int, int, int],
    device,
):
    """
    Returns center locations of each feature cell in (x, y, z) order.

    width, height, depth: spatial size of the feature map
    stride: (sx, sy, sz) stride w.r.t input image
    """

    sx, sy, sz = stride

    shifts_x = torch.arange(
        0, width * sx, step=sx,
        dtype=torch.float32, device=device
    )
    shifts_y = torch.arange(
        0, height * sy, step=sy,
        dtype=torch.float32, device=device
    )
    shifts_z = torch.arange(
        0, depth * sz, step=sz,
        dtype=torch.float32, device=device
    )

    shift_x, shift_y, shift_z = torch.meshgrid(
        shifts_x, shifts_y, shifts_z, indexing="ij"
    )

    locations = torch.stack(
        (shift_x.reshape(-1), shift_y.reshape(-1), shift_z.reshape(-1)),
        dim=1,
    )

    # add half-stride offset to get center of each cell
    center_offset = torch.tensor(
        [sx / 2, sy / 2, sz / 2],
        dtype=torch.float32,
        device=device,
    )

    locations = locations + center_offset

    return locations


class AnisotropicATSSMatcher(Matcher):
    def __init__(
        self,
        num_candidates: int = 4,
        similarity_fn: Callable[[Tensor, Tensor], Tensor] = box_iou,  # type: ignore
        center_in_gt: bool = True,
        threshold_strategy: Literal["default", "topk", "mean"] = "default",
        debug: bool = False,
    ):
        """
        Compute matching based on ATSS https://arxiv.org/abs/1912.02424
        `Bridging the Gap Between Anchor-based and Anchor-free Detection
        via Adaptive Training Sample Selection`

        Args:
            num_candidates: number of positions to select candidates from.
                Smaller value will result in a higher matcher threshold and less matched candidates.
            similarity_fn: function for similarity computation between boxes and anchors
            center_in_gt: If False (default), matched anchor center points do not need
                to lie withing the ground truth box. Recommend False for small objects.
                If True, will result in a strict matcher and less matched candidates.
            debug: if True, will print the matcher threshold in order to
                tune ``num_candidates`` and ``center_in_gt``.
        """
        super().__init__(similarity_fn=similarity_fn)
        self.num_candidates = num_candidates
        self.min_dist = 0.01
        self.center_in_gt = center_in_gt
        self.threshold_strategy = threshold_strategy
        self.debug = debug
        logging.info(
            f"Running ATSS Matching with num_candidates={self.num_candidates} and center_in_gt {self.center_in_gt}."
        )

    def compute_matches(
        self, boxes: Tensor, anchors: Tensor, num_anchors_per_level: Sequence[int], num_anchors_per_loc: int,
            spacing: Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Compute matches according to ATTS for a single image
        Adapted from
        (https://github.com/sfzhang15/ATSS/blob/79dfb28bd1/atss_core/modeling/rpn/atss/loss.py#L180-L184)
        """
        num_gt = boxes.shape[0]
        num_anchors = anchors.shape[0]

        distances_, _, anchors_center = boxes_center_distance(boxes, anchors)  # num_boxes x anchors
        distances = convert_to_tensor(distances_)

        # select candidates based on center distance
        candidate_idx_list = []
        start_idx = 0
        for _, apl in enumerate(num_anchors_per_level):
            end_idx = start_idx + apl * num_anchors_per_loc

            # topk: total number of candidates per position
            topk = min(self.num_candidates * num_anchors_per_loc, apl)
            # torch.topk() does not support float16 cpu, need conversion to float32 or float64
            _, idx = distances[:, start_idx:end_idx].to(torch.float32).topk(topk, dim=1, largest=False)
            # idx: shape [num_boxes x topk]
            candidate_idx_list.append(idx + start_idx)

            start_idx = end_idx
        # [num_boxes x num_candidates] (index of candidate anchors)
        candidate_idx = torch.cat(candidate_idx_list, dim=1)

        match_quality_matrix = self.similarity_fn(boxes, anchors)  # [num_boxes x anchors]
        candidate_ious = match_quality_matrix.gather(1, candidate_idx)  # [num_boxes, n_candidates]

        # corner case, n_candidates<=1 will make iou_std_per_gt NaN
        if candidate_idx.shape[1] <= 1:
            matches = -1 * torch.ones((num_anchors,), dtype=torch.long, device=boxes.device)
            matches[candidate_idx] = 0
            return match_quality_matrix, matches

        # compute adaptive iou threshold
        iou_mean_per_gt = candidate_ious.mean(dim=1)  # [num_boxes]

        if self.threshold_strategy == "mean":
            iou_thresh_per_gt = iou_mean_per_gt  # [num_boxes]
            is_pos = candidate_ious >= iou_thresh_per_gt[:, None]  # [num_boxes x n_candidates]
        elif self.threshold_strategy == "topk":
            # Get the top-k IoUs per ground-truth box
            topk_values, _ = torch.topk(candidate_ious, self.num_candidates, dim=1)  # [num_boxes, k]
            # Set is_pos based on whether IoUs are in the top-k
            is_pos = candidate_ious >= topk_values[:, -1, None]  # [num_boxes, n_candidates]
            iou_thresh_per_gt = None
        else:
            iou_std_per_gt = candidate_ious.std(dim=1)  # [num_boxes]
            iou_thresh_per_gt = iou_mean_per_gt + iou_std_per_gt  # [num_boxes]
            is_pos = candidate_ious >= iou_thresh_per_gt[:, None]  # [num_boxes x n_candidates]

        if self.debug:
            print(f"Anchor matcher threshold: {iou_thresh_per_gt}")

        if self.center_in_gt:  # can discard all candidates in case of very small objects :/
            # center point of selected anchors needs to lie within the ground truth
            boxes_idx = (
                torch.arange(num_gt, device=boxes.device, dtype=torch.long)[:, None]
                .expand_as(candidate_idx)
                .contiguous()
            )  # [num_boxes x n_candidates]
            is_in_gt_ = centers_in_boxes(
                anchors_center[candidate_idx.view(-1)], boxes[boxes_idx.view(-1)], eps=self.min_dist
            )
            is_in_gt = convert_to_tensor(is_in_gt_)
            is_pos = is_pos & is_in_gt.view_as(is_pos)  # [num_boxes x n_candidates]

        # in case on anchor is assigned to multiple boxes, use box with highest IoU
        for ng in range(num_gt):
            candidate_idx[ng, :] += ng * num_anchors
        ious_inf = torch.full_like(match_quality_matrix, -INF).view(-1)
        index = candidate_idx.view(-1)[is_pos.view(-1)]
        ious_inf[index] = match_quality_matrix.view(-1)[index]
        ious_inf = ious_inf.view_as(match_quality_matrix)

        matched_vals, matches = ious_inf.to(COMPUTE_DTYPE).max(dim=0)
        matches[matched_vals == -INF] = self.BELOW_LOW_THRESHOLD
        return match_quality_matrix, matches


def generate_3d_anchors(image_size, feature_shapes, anchor_sizes, device='cpu'):
    """
    Generate 3D anchors for each scale.
    """
    assert len(feature_shapes) == len(anchor_sizes), "Number of feature shapes must match number of anchor sizes."

    anchors_all_scales = []
    W, H, D = image_size[-3:]

    # For each scale:
    for F, A in zip(feature_shapes, anchor_sizes):
        W_i, H_i, D_i = F[-3:]
        a_w, a_h, a_d = A[-3:]
        # Compute downsampling factors
        stride_w = W / float(W_i)
        stride_h = H / float(H_i)
        stride_d = D / float(D_i)

        # Generate a grid for the feature map indices
        # h_idx: [0, 1, ..., H_i-1], etc.
        w_idx = torch.arange(W_i, dtype=torch.float32)
        h_idx = torch.arange(H_i, dtype=torch.float32)
        d_idx = torch.arange(D_i, dtype=torch.float32)

        # Create a meshgrid (H_i x W_i x D_i) for the coordinates
        # These are the indices in the feature space
        w_grid, h_grid, d_grid = torch.meshgrid(w_idx, h_idx, d_idx, indexing='ij')

        # Compute center coordinates in original image space
        # The center of the voxel (h,w,d) maps roughly to:
        center_x = (w_grid + 0.5) * stride_w
        center_y = (h_grid + 0.5) * stride_h
        center_z = (d_grid + 0.5) * stride_d
        # center_x = w_grid * stride_w
        # center_y = h_grid * stride_h
        # center_z = d_grid * stride_d

        # Half sizes of the anchor
        half_w = a_w / 2.0
        half_h = a_h / 2.0
        half_d = a_d / 2.0

        # Compute the anchor box coordinates
        x1 = center_x - half_w
        y1 = center_y - half_h
        z1 = center_z - half_d
        x2 = center_x + half_w
        y2 = center_y + half_h
        z2 = center_z + half_d

        # Stack them into shape (H_i*W_i*D_i, 6)
        anchors = torch.stack([x1, y1, z1, x2, y2, z2], dim=-1)  # [H_i, W_i, D_i, 6]
        anchors = anchors.reshape(-1, 6)

        anchors_all_scales.append(anchors.to(device=device))

    return anchors_all_scales
