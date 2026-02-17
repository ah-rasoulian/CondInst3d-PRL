from collections import OrderedDict

from matplotlib import pyplot as plt

from condinst3d.visualization.utils import simple_binary_to_gray
from condinst3d.visualization.binary_seg_visualizer import BinarySegSliceVisualizer
from condinst3d.visualization.instance_seg_visualizer import InstanceSegSliceVisualizer


class ListInstanceBoxSegSliceVisualizer(InstanceSegSliceVisualizer):

    def _get_fig(self, nrows, ncols):
        fig, axs = plt.subplots(
            nrows, ncols, figsize=self._get_figsize(nrows, ncols, 0),
            sharey=True, sharex=True, squeeze=False,
            gridspec_kw=dict(hspace=0.0, wspace=0.0)
        )
        return fig, axs

    def _close_fig(self, fig):
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.0, wspace=0.0)
        plt.close()
        return fig

    def plot(self, inputs, y_pred, y_true, stats, add_info_text=False,
             title=None, boxes_pred=None, boxes_true=None, boxes_scores=None):
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
        figs = OrderedDict()
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
            figs["empty"] = binary_visualizer.plot(inputs, y_pred, y_true, title)
            return figs

        nrows = len(self.show_slices)

        if self.pred_seg_is_binary:
            y_pred = simple_binary_to_gray(y_pred)

        if n_tps > 0:
            for i in range(n_tps):
                fig, axs = self._get_fig(nrows, ncols)
                self._plot_one_instance(
                    inputs, y_true, y_pred, boxes_true, boxes_pred, boxes_scores=boxes_scores,
                    ids=tp_ids[i], error_type="tp", map_ids=tp_map_ids[i], row_index=0, axs=axs)
                if title:
                    self._plot_title(fig, title + " / TP {}".format(tp_ids[i]))
                if add_info_text:
                    info = ""
                    iou = stats['y_pred'][tp_map_ids[i][1]].get('iou', None)
                    if iou is not None:
                        info += f"IoU={iou[0]:.2f}\n"
                    score = stats['y_pred'][tp_map_ids[i][1]].get('score', None)
                    if score is not None:
                        info += f"Score={score:.2f}\n]"
                    self._add_text_top_right(fig, info)
                fig = self._close_fig(fig)
                figs["tp{}".format(tp_ids[i])] = fig

        if n_fns:
            for i in range(n_fns):
                fig, axs = self._get_fig(nrows, ncols)
                self._plot_one_instance(
                    inputs, y_true, y_pred, boxes_true, boxes_pred, boxes_scores=boxes_scores,
                    ids=fn_ids[i], error_type="fn", map_ids=fn_map_ids[i], row_index=0, axs=axs)
                if title:
                    self._plot_title(fig, title + " / FN {}".format(fn_ids[i]))
                fig = self._close_fig(fig)
                figs["fn{}".format(fn_ids[i])] = fig

        if n_fps:
            for i in range(n_fps):
                fig, axs = self._get_fig(nrows, ncols)
                self._plot_one_instance(
                    inputs, y_true, y_pred, boxes_true, boxes_pred, boxes_scores=boxes_scores,
                    ids=fp_ids[i], error_type="fp", map_ids=fp_map_ids[i], row_index=0, axs=axs)
                if title:
                    self._plot_title(fig, title + " / FP {}".format(fp_ids[i]))
                if add_info_text:
                    info = f"Score={stats['y_pred'][fp_map_ids[i][1]]['score']:.2f}"
                    self._add_text_top_right(fig, info)
                fig = self._close_fig(fig)
                figs["fp{}".format(fp_ids[i])] = fig

        return figs
