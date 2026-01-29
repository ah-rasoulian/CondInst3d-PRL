from math import floor, ceil

import numpy as np
import torch
from matplotlib import pyplot as plt, patches
from skimage.morphology import dilation

from monai.apps.detection.transforms.array import SpatialCropBox
from monai.apps.detection.transforms.box_ops import convert_box_to_mask, convert_mask_to_box
from monai.transforms import CenterSpatialCrop, SpatialCrop
from monai.utils import convert_to_numpy
from condinst3d.visualization.binary_seg_visualizer import BinarySegSliceVisualizer
from condinst3d.visualization.utils import count_error_type, simple_binary_to_gray


class InstanceSegSliceVisualizer(BinarySegSliceVisualizer):
    segmentation_fg = 1
    segmentation_other = 3
    boxes_fg = 3

    def __init__(self, crop_size = (64, 64),
                 pred_seg_is_binary: bool = True,
                 draw_boxes: bool = True,
                 *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pred_seg_is_binary = pred_seg_is_binary
        self.crop_size = crop_size
        self.draw_boxes = draw_boxes

    def _get_fontsize(self):
        return 15 * self.figsize / 3.0

    def _get_nrows(self, n_instances):
        return n_instances * len(self.show_slices)
        return nrows

    def _lesions_stats_to_errors(self, stats):
        ec = count_error_type(stats=stats)
        return ec["TP"], ec["FN"], ec["FP"]

    def _get_errors_ids(self, stats):
        fn_ids = list(stats['y_true'].keys())
        fp_ids = list([k for k, rid in stats['y_pred'].items() if rid["type"] == "FP"])
        tp_ids = list([rid["regions"][0] for rid in stats['y_pred'].values() if rid["type"] == "TP"])
        return tp_ids, fn_ids, fp_ids

    def _get_mapping_ids(self, stats):
        # same order as get_error_ids
        fn_ids = [(k, None) for k in stats['y_true'].keys()]
        fp_ids = [(None, k) for k, rid in stats['y_pred'].items() if rid["type"] == "FP"]
        tp_ids = [(rid["regions"][0], k) for k, rid in stats['y_pred'].items() if rid["type"] == "TP"]
        # (idx gt, idx pred) idx is none if no mapping
        return tp_ids, fn_ids, fp_ids

    def _get_slice_index(self, y_true, y_pred, ids, error_type, axis=None):
        if axis is None:
            axis = self.inplane_axis
        if error_type.lower() == "tp":
            islice = np.argmax(np.sum(y_true == ids, axis=axis)).item()
        elif error_type.lower() == "fn":
            islice = np.argmax(np.sum(y_true == ids, axis=axis)).item()
        elif error_type.lower() == "fp":
            islice = np.argmax(np.sum(y_pred == ids, axis=axis)).item()
        else:
            raise ValueError("unknown error type {}".format(error_type))
        return islice

    def _filter_boxes(self, boxes, z_index):
        ans = []
        for bbox_np in boxes:
            bbox = np.round(bbox_np).astype(int).tolist()
            if bbox[2] <= z_index <= bbox[5]:
                ans.append(bbox_np)
        if len(ans):
            return np.stack(ans, axis=0)
        return None

    def _convert_boxes_to_mask(self, boxes, shape):
        if boxes is None:
            return None
        else:
            mask = convert_box_to_mask(
                boxes, np.ones((boxes.shape[0],)),
                spatial_size=shape, bg_label=0)
            #mask = np.sum(mask, axis=(0,))
            #mask = dilation(mask) - mask
            #mask = np.where(mask > 0, self.boxes_fg, 0)
        return mask

    def _get_sliced_boxes(self, boxes, zindex, scores=None):
        if boxes is None:
            return None, None
        sliced_ans = []
        slices_scores = []
        for i, bbox in enumerate(boxes):
            if not (floor(bbox[2]) <= zindex <= ceil(bbox[5])):
                continue
            sliced_ans.append(bbox)
            if scores is not None:
                slices_scores.append(scores[i])
        if len(sliced_ans) == 0:
            return None, None
        slices_ans = np.stack(sliced_ans, axis=0)
        if scores is None:
            slices_scores = None
        else:
            slices_scores = np.stack(slices_scores, axis=0)
        return slices_ans, slices_scores

    def _plot_segmentation_slices(self, y_true, y_pred, boxes_true, boxes_pred,
                                  z_index, axs, row_index=0,
                                  boxes_scores=None):
        seg_cmap = self._get_seg_colormap("black")
        seg_norm = self._get_seg_norm()
        axs[0, -1].set_title("y_pred")
        axs[0, -2].set_title("y_true")
        zmax = self._get_max_z(y_true)
        for islice, s in enumerate(self.show_slices):
            cs = z_index + s
            cs = self._clip_z(cs, zmax)
            cs_y_true = self._get_masked_segmentation(self._get_slice(y_true, cs))
            cs_y_pred = self._get_masked_segmentation(self._get_slice(y_pred, cs))
            axs[row_index + islice, -2].imshow(
                cs_y_true, cmap=seg_cmap, norm=seg_norm,
                interpolation=self.label_interpolation)
            axs[row_index + islice, -1].imshow(
                cs_y_pred, cmap=seg_cmap, norm=seg_norm,
                interpolation=self.label_interpolation)
            if boxes_pred is not None:
                cs_boxes_pred, cs_boxes_scores = self._get_sliced_boxes(boxes_pred, cs, scores=boxes_scores)
                ax = axs[row_index + islice, -1]
                if cs_boxes_pred is not None:
                    for i, bbox in enumerate(cs_boxes_pred):
                        if self.draw_boxes:
                            rect = patches.Rectangle(
                                (bbox[1], bbox[0]), bbox[4] - bbox[1], bbox[3] - bbox[0],
                                linewidth=1, edgecolor='r', facecolor='none'
                            )
                            # rect = patches.Rectangle(
                            #     (bbox[0], bbox[1]), bbox[3] - bbox[0], bbox[4] - bbox[1],
                            #     linewidth=1, edgecolor='r', facecolor='none'
                            # )
                            ax.add_patch(rect)
                        if cs_boxes_scores is not None:
                            score = cs_boxes_scores[i]
                            ax.annotate(
                                text="score={:.2f}".format(score),
                                xy=(bbox[1], bbox[0]), color='r', ha='center',
                                fontsize=self._get_fontsize(), xytext=(-1, 3), textcoords='offset points')
                #cs_boxes_pred = self._get_masked_segmentation(self._get_slice(boxes_pred, cs))
                # axs[row_index + islice, -1].imshow(
                #     cs_boxes_pred, cmap=transparent_cmap,
                #     interpolation=self.label_interpolation
                # )

            if boxes_true is not None:
                cs_boxes_true, _ = self._get_sliced_boxes(boxes_true, cs)
                if cs_boxes_true is not None:
                    for bbox in cs_boxes_true:
                        rect = patches.Rectangle(
                            (bbox[1], bbox[0]), bbox[4] - bbox[1], bbox[3] - bbox[0],
                            linewidth=1, edgecolor='r', facecolor='none'
                        )
                        axs[row_index + islice, -1].add_path(rect)
                # cs_boxes_true = self._get_masked_segmentation(self._get_slice(boxes_true, cs))
                # axs[row_index + islice, -1].imshow(
                #    cs_boxes_true, cmap=transparent_cmap,
                #     interpolation=self.label_interpolation
                # )

    def _get_roi_center(self, y_true, y_pred, ids, error_type):
        error_type = error_type.lower()
        if error_type in {"tp", "fn"}:
            y = y_true
        elif error_type == "fp":
            y = y_pred
        else:
            raise ValueError("unknown error type {}".format(error_type))
        if isinstance(ids, torch.Tensor):
            ids = ids.numpy()
        indices = np.where(y == ids)
        xmin, xmax = indices[0].min(), indices[0].max()
        ymin, ymax = indices[1].min(), indices[1].max()
        zmin, zmax = indices[2].min(), indices[2].max()
        xcenter = xmax - (xmax - xmin) // 2
        ycenter = ymax - (ymax - ymin) // 2
        zcenter = zmax - (zmax - zmin) // 2
        return xcenter, ycenter, zcenter

    def _get_boxes_cropper(self, roi_center):
        return SpatialCropBox(
            roi_center=roi_center,
            roi_size=list(self.crop_size) + [len(self.show_slices)]
        )
    def _crop_boxes(self, boxes, cropper, scores=None):
        if boxes is None:
            return None, None
        if scores is None:
            labels = np.ones((boxes.shape[0],))
        else:
            labels = scores
        cropped_boxes, cropped_scores = cropper(boxes, labels)
        if cropped_boxes.shape[0] == 0 or scores is None:
            cropped_boxes = None
            cropped_scores = None
        return cropped_boxes, cropped_scores

    def _get_image_cropper(self, roi_center):
        cropper = SpatialCrop(
            roi_center=roi_center,
            roi_size=list(self.crop_size) + [len(self.show_slices)]
        )
        return cropper

    def _crop_array(self, array, cropper):
        if array is None:
            return None
        array = array[None,]
        return cropper(array)[0]

    def _get_highlighted_segmentation(self, y, ids):
        y_recolored = np.zeros(y.shape, dtype=np.uint8)
        if ids == None:
            other_idx = np.where(y > 0)
        else:
            fg_idx = np.where(y == ids)
            other_idx = np.where(np.logical_and(y > 0, y != ids))
            y_recolored[fg_idx] = self.segmentation_fg

        y_recolored[other_idx] = self.segmentation_other
        return y_recolored

    def _plot_one_instance(self, inputs, y_true, y_pred, boxes_true, boxes_pred,
                           ids, error_type, map_ids, row_index, axs, boxes_scores=None):
        #islice = self._get_slice_index(y_true, y_pred, ids=ids, error_type=error_type.lower())
        xcenter, ycenter, zcenter = self._get_roi_center(y_true, y_pred, ids=ids, error_type=error_type.lower())
        image_cropper = self._get_image_cropper((xcenter, ycenter, zcenter))
        boxes_cropper = self._get_boxes_cropper((xcenter, ycenter, zcenter))
        cropped_z_index = len(self.show_slices) // 2

        for icol, ichannel in enumerate(self.channels_order):
            channel_name = self.img_channels_names[ichannel]
            norm = self._get_intensity_normalization(inputs[ichannel], channel_name)
            self._plot_image_slices(
                image=self._crop_array(inputs[ichannel], image_cropper),
                col_index=icol,
                z_index=cropped_z_index,
                title=channel_name,
                axs=axs, norm=norm,
                row_index=row_index,
            )
            if self.channel_seg_under_image and ichannel == self.channel_seg_under_image:
                for col in (-2, -1):
                    self._plot_image_slices(
                        image=self._crop_array(inputs[ichannel], image_cropper),
                        col_index=col,
                        z_index=cropped_z_index,
                        title=channel_name,
                        axs=axs, norm=norm,
                        row_index=row_index,
                    )
        if map_ids[1] is None or boxes_pred is None or boxes_true is None:
            boxes_pred = None
            boxes_scores = None
        else:
            boxes_pred = boxes_pred[map_ids[1] - 1 : map_ids[1]]
            boxes_scores = boxes_scores[map_ids[1] - 1 : map_ids[1]]
        cropped_boxes_pred, cropped_boxes_scores = self._crop_boxes(boxes_pred, boxes_cropper, scores=boxes_scores)
        cropped_boxes_true, _ = self._crop_boxes(boxes_true, boxes_cropper)
        self._plot_segmentation_slices(
            self._get_highlighted_segmentation(self._crop_array(y_true, image_cropper), ids=map_ids[0]),
            self._get_highlighted_segmentation(self._crop_array(y_pred, image_cropper), ids=map_ids[1]),
            cropped_boxes_true,
            cropped_boxes_pred, boxes_scores=cropped_boxes_scores,
            z_index=cropped_z_index, axs=axs, row_index=row_index)
        for ax in axs.flatten():
            ax.set_xticks([])
            ax.set_yticks([])

    def _ensure_type_boxes(self, arr):
        if arr is not None:
            if arr.dtype == torch.bfloat16:
                arr = arr.type(torch.float16)
            arr = convert_to_numpy(arr, dtype="float")
        return arr

    def _get_figsize(self, nrows, ncols, hspace):
        ratio = self.crop_size[0] / self.crop_size[1]
        return self.figsize * ratio * ncols, self.figsize * ratio * nrows #* (1+hspace)


    def _check_inputs(self, inputs, y_pred, y_true, stats, boxes_pred, boxes_true, boxes_scores):
        inputs = self._ensure_type_images(inputs)
        y_true = self._ensure_type_segmentation(y_true)
        y_pred = self._ensure_type_segmentation(y_pred)
        if boxes_pred is not None:
            boxes_pred = self._ensure_type_boxes(boxes_pred)
        if boxes_true is not None:
            boxes_true = self._ensure_type_boxes(boxes_true)
        if boxes_scores is not None:
            boxes_scores = self._ensure_type_boxes(boxes_scores)

        # for now support of one class only
        stats = stats[0]
        if y_true.ndim == 4:
            y_true = y_true[0]
        if y_pred.ndim == 4:
            y_pred = y_pred[0]
        assert y_pred.ndim == 3
        assert y_true.ndim == 3
        assert inputs.ndim == 4
        return inputs, y_pred, y_true, stats, boxes_pred, boxes_true, boxes_scores

    def plot(self, inputs, y_pred, y_true, stats, title=None, boxes_pred=None, boxes_true=None, boxes_scores=None):
        inputs, y_pred, y_true, stats, boxes_pred, boxes_true, boxes_scores = self._check_inputs(
            inputs, y_pred, y_true, stats, boxes_pred, boxes_true, boxes_scores
        )

        if stats is not None:
            tp_ids, fn_ids, fp_ids = self._get_errors_ids(stats)
            n_tps, n_fns, n_fps = self._lesions_stats_to_errors(stats)
            tp_map_ids, fn_map_ids, fp_map_ids = self._get_mapping_ids(stats)
        else:
            n_tps, n_fns, n_fps = 0, 0, 0

        ncols = self._get_ncols(inputs, y_true, y_pred)

        n_instances = sum([n_tps, n_fns, n_fps])
        if n_instances == 0:
            binary_visualizer = BinarySegSliceVisualizer(
                image_interpolation=self.image_interpolation,
                label_interpolation=self.label_interpolation,
                inplane_axis=self.inplane_axis,
                outplane_axis=self.outplane_axis,
                user_cslice=self.user_cslice,
                show_slices=(0, ),
                img_channels=self.img_channels_names,
                seg_channels=self.seg_channels_names,
                figsize=self.figsize,
                range_percentiles=self.range_percentiles,
                channel_seg_under_image=self.channel_seg_under_image
            )
            return binary_visualizer.plot(inputs, y_pred, y_true, title)
        elif n_instances == 1:
            nrows = len(self.show_slices)
            fig, axs = plt.subplots(
                nrows, ncols, figsize=self._get_figsize(nrows, ncols, 0),
                sharey=True, sharex=True, squeeze=False,
                gridspec_kw=dict(hspace=0.0, wspace=0.0)
            )
            subfigs = None
        else:
            total_nrows = self._get_nrows(n_tps) + self._get_nrows(n_fns) + self._get_nrows(n_fps)
            hspace = 0.1
            fig = plt.figure(
                constrained_layout=False, tight_layout=False,
                figsize=self._get_figsize(total_nrows, ncols, hspace),
            )
            subfigs = fig.subfigures(n_instances, 1, hspace=hspace, squeeze=False)

        if self.pred_seg_is_binary:
            y_pred = simple_binary_to_gray(y_pred)

        if title:
            self._plot_title(fig, title)

        subfig_i = 0
        if n_tps > 0:
            for i in range(n_tps):
                if subfigs is not None:
                    axs = subfigs[subfig_i, 0].subplots(
                        len(self.show_slices), ncols, squeeze=False,
                        sharex=True, sharey=True,
                        gridspec_kw=dict(hspace=0.0, wspace=0.0)
                    )
                    subfigs[subfig_i, 0].suptitle("TP {}".format(tp_ids[i]), fontweight="bold")
                    subfig_i += 1
                self._plot_one_instance(
                    inputs, y_true, y_pred, boxes_true, boxes_pred, boxes_scores=boxes_scores,
                    ids=tp_ids[i], error_type="tp", map_ids=tp_map_ids[i], row_index=0, axs=axs)

        if n_fns:
            for i in range(n_fns):
                if subfigs is not None:
                    axs = subfigs[subfig_i, 0].subplots(
                        len(self.show_slices), ncols, squeeze=False,
                        sharex=True, sharey=True,
                        gridspec_kw=dict(hspace=0.0, wspace=0.0)
                    )
                    subfigs[subfig_i, 0].suptitle("FN {}".format(fn_ids[i]), fontweight="bold")
                    subfig_i += 1
                self._plot_one_instance(
                    inputs, y_true, y_pred, boxes_true, boxes_pred, boxes_scores=boxes_scores,
                    ids=fn_ids[i], error_type="fn", map_ids=fn_map_ids[i], row_index=0, axs=axs)

        if n_fps:
            for i in range(n_fps):
                if subfigs is not None:
                    axs = subfigs[subfig_i, 0].subplots(
                        len(self.show_slices), ncols, squeeze=False,
                        sharex=True, sharey=True,
                        gridspec_kw=dict(hspace=0.0, wspace=0.0)
                    )
                    subfigs[subfig_i, 0].suptitle("FP {}".format(fp_ids[i]), fontweight="bold")
                    subfig_i += 1
                self._plot_one_instance(
                    inputs, y_true, y_pred, boxes_true, boxes_pred, boxes_scores=boxes_scores,
                    ids=fp_ids[i], error_type="fp", map_ids=fp_map_ids[i], row_index=0, axs=axs)

        plt.subplots_adjust(wspace=0.0, hspace=0.0)
        plt.close()
        return fig
