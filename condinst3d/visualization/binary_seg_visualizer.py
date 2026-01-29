import copy
import os

import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt

from monai.utils import convert_to_numpy


def write_fig_to_file(fig, title, rootdir=None):
    basename = "{}.png".format(title.replace(".mnc.gz", ""))
    if rootdir is not None:
        path = os.path.join(rootdir, basename)
    else:
        path = basename
    try:
        os.makedirs(os.path.dirname(path))
    except Exception:
        pass
    fig.savefig(path)


class BinarySegSliceVisualizer:
    def __init__(self,
                 image_interpolation="antialiased",
                 label_interpolation="nearest",
                 inplane_axis=(-3, -2),
                 outplane_axis=-1,
                 user_cslice=None,
                 show_slices=(0,),
                 img_channels=None,
                 seg_channels=None,
                 figsize=3,
                 range_percentiles=None,
                 channel_seg_under_image=None,
                 channels_order=None,
                 transpose_axes=None,
                 flip_axes=None
                 ):
        self.image_interpolation = image_interpolation
        self.label_interpolation = label_interpolation
        self.inplane_axis = inplane_axis
        self.outplane_axis = outplane_axis
        self.user_cslice = user_cslice
        self.show_slices = show_slices
        self.img_channels_names = img_channels
        self.seg_channels_names = seg_channels

        self.figsize = figsize
        self.channel_seg_under_image = channel_seg_under_image
        if channels_order is None:
            self.channels_order = range(len(img_channels))
        else:
            self.channels_order = [img_channels.index(c) for c in channels_order]
        if range_percentiles is None:
            self.range_percentiles = {}
        else:
            self.range_percentiles = range_percentiles
        self.transpose_axes = transpose_axes
        self.flip_axes = flip_axes

    def _get_sum_outplane(self, y):
        return np.sum(y > 0, axis=self.inplane_axis)

    def _get_max_z(self, img):
        return img.shape[self.outplane_axis]

    def _get_slice_index(self, y_true, y_pred):
        #return int(y_true.shape[self.outplane_axis] // 2)

        # ncols_seg = 0
        # if user_cslice < 1:
        #     # select user select slice
        #     cslice = int(float(user_cslice) * nz)
        # else:
        #     cslice = nz // 2

        nz = y_true.shape[self.outplane_axis]
        seg_slices = self._get_sum_outplane(y_true)
        pred_slices = self._get_sum_outplane(y_pred)
        if np.sum(seg_slices).item() < 1.0:
            # no ground truth, fall back slice selection in prediction
            pred_sum = np.sum(pred_slices).item()
            if pred_sum < 1.0:
                if self.user_cslice is not None and self.user_cslice < 1:
                    # select user select slice
                    cslice = int(float(self.user_cslice) * nz)
                else:
                    cslice = nz // 2
            else:
                cslice = np.argmax(pred_slices).item()
                # argmax_slices = np.argmax(pred_slices, axis=1).tolist()
                # if argmax_slices[0] != 0:
                #     # give gvf priority over ct2f
                #     cslice = argmax_slices[0]
                # else:
                #     cslice = argmax_slices[1]
        else:
            # select slice with max lesion voxels
            cslice = np.argmax(seg_slices).item()
            # argmax_slices = np.argmax(seg_slices, axis=1).tolist()
            # if argmax_slices[0] != 0:
            #     # give gvf priority over ct2f
            #     cslice = argmax_slices[0]
            # else:
            #     cslice = argmax_slices[1]
        return cslice

    def _get_ncols(self, inputs, y_true, y_pred):
        ncols = inputs.shape[0]
        if y_true is not None:
            ncols += 1
        if y_pred is not None:
            ncols += 1
        return ncols

    def _get_nrows(self):
        nrows = len(self.show_slices)
        return nrows

    def _clip_z(self, z, max_z):
        z = max(0, z)
        z = min(z, max_z - 1)
        return z

    def _get_slice_size_ratio(self, img_shape):
        slice_shape = [img_shape[i] for i in self.inplane_axis]
        ratio = slice_shape[0] / slice_shape[1]
        return ratio

    def _get_figsize(self, img_shape, nrows, ncols):
        ratio = self._get_slice_size_ratio(img_shape)
        return self.figsize * ratio * ncols, self.figsize * ratio * nrows

    def _get_seg_colormap(self, color="black"):
        if self.channel_seg_under_image is None:
            alpha = 1.0
        else:
            alpha = 0.0
        vmax = 9
        cmap = copy.copy(plt.cm.get_cmap("Set1", vmax - 1))
        cmap.set_bad(color, alpha=alpha)
        return cmap

    def _get_seg_norm(self):
        return matplotlib.colors.Normalize(vmin=1, vmax=9)

    def _get_masked_segmentation(self, y):
        return np.ma.masked_where(y == 0, y)

    def _get_slice(self, img, z_index):
        #self.outplane_axis =
        return img[:, :, z_index]

    def _get_intensity_normalization(self, img, title):
        # if freqmap_range_per:
        #         vmin = np.percentile(ix[ichannel], q=freqmap_range_per[0])
        #         vmax = np.percentile(ix[ichannel], q=freqmap_range_per[1])
        #     else:
        if title in self.range_percentiles:
            vmin = np.percentile(img, q=self.range_percentiles[title][0])
            vmax = np.percentile(img, q=self.range_percentiles[title][1])
            norm = matplotlib.colors.PowerNorm(gamma=1, vmin=vmin, vmax=vmax)
        else:
            norm = None
        return norm

    def _plot_title(self, fig, title):
        fig.suptitle(title, y=0.99)

    def _add_text_top_right(self, fig, text, fontsize=10):
        fig.text(
            0.9, 1.0,
            text,
            ha='right',
            va='top',
            fontsize=fontsize,
        )

    def _plot_ylabel(self, z_index, row_index, axs):
        axs[row_index, 0].set_ylabel(f"slice {z_index}")

    def _plot_image_slices(self, image, title, col_index, z_index, axs, row_index=0, norm=None):
        zmax = self._get_max_z(image)
        axs[0, col_index].set_title(f"{title}")
        for islice, s in enumerate(self.show_slices):
            cs = z_index + s
            cs = self._clip_z(cs, zmax)
            cs_inputs = self._get_slice(image, cs)
            axs[islice + row_index, col_index].imshow(
                cs_inputs, cmap="gray",
                interpolation=self.image_interpolation, norm=norm, vmin=min(0, image.min()), vmax=max(1, image.max()),
            )
            axs[islice + row_index, 0].set_ylabel("slice {}".format(cs))

    def _plot_segmentation_slices(self, y_true, y_pred, z_index, axs):
        seg_cmap = self._get_seg_colormap()
        seg_norm = self._get_seg_norm()
        axs[0, -2].set_title("y_true")
        axs[0, -1].set_title("y_pred")
        zmax = self._get_max_z(y_true)
        for islice, s in enumerate(self.show_slices):
            cs = z_index + s
            cs = self._clip_z(cs, zmax)
            cs_y_true = self._get_masked_segmentation(self._get_slice(y_true, cs))
            cs_y_pred = self._get_masked_segmentation(self._get_slice(y_pred, cs))

            axs[islice, -2].imshow(
                cs_y_true, cmap=seg_cmap, norm=seg_norm,
                interpolation=self.label_interpolation)
            axs[islice, -1].imshow(
                cs_y_pred, cmap=seg_cmap, norm=seg_norm,
                interpolation=self.label_interpolation)

    def _ensure_type_images(self, arr):
        return convert_to_numpy(arr, dtype="float")

    def _ensure_type_segmentation(self, arr):
        return convert_to_numpy(arr, dtype="int")

    def maybe_transpose(self, *args):
        if not self.transpose_axes and not self.flip_axes:
            return args
        ret = []
        for arr in args:
            new_arr = arr
            if self.transpose_axes:
                new_arr = np.transpose(new_arr, self.transpose_axes)
            if self.flip_axes:
                new_arr = np.flip(new_arr, self.flip_axes)
            ret.append(new_arr)
        return ret

    def plot(self, inputs, y_pred, y_true, title=None):
        inputs = self._ensure_type_images(inputs)
        y_true = self._ensure_type_segmentation(y_true)
        y_pred = self._ensure_type_segmentation(y_pred)

        inputs, y_true, y_pred = self.maybe_transpose(inputs, y_true, y_pred)

        if y_true.ndim == 4:
            y_true = y_true[0]
        if y_pred.ndim == 4:
            y_pred = y_pred[0]
        assert y_pred.ndim == 3
        assert y_true.ndim == 3
        assert inputs.ndim == 4

        cslice = self._get_slice_index(y_true, y_pred)
        nrows = self._get_nrows()
        ncols = self._get_ncols(inputs, y_true, y_pred)
        fig, axs = plt.subplots(
            nrows, ncols, squeeze=False,
            figsize=self._get_figsize(y_true.shape, nrows, ncols),
            sharey=True, sharex=True
        )

        if title:
            self._plot_title(fig, title)

        for icol, ichannel in enumerate(self.channels_order):
            channel_name = self.img_channels_names[ichannel]
            norm = self._get_intensity_normalization(inputs[ichannel], title)
            self._plot_image_slices(
                image=inputs[ichannel],
                col_index=icol,
                z_index=cslice,
                title=channel_name,
                axs=axs, norm=norm
            )
            if self.channel_seg_under_image is not None and ichannel == self.channel_seg_under_image:
                for col in (-2, -1):
                    self._plot_image_slices(
                        image=inputs[ichannel],
                        col_index=col,
                        z_index=cslice,
                        title=channel_name,
                        axs=axs, norm=norm,
                    )
        self._plot_segmentation_slices(
            y_true, y_pred, z_index=cslice, axs=axs
        )

        self._plot_ylabel(z_index=cslice, row_index=nrows // 2, axs=axs)
        fig.tight_layout()
        plt.subplots_adjust(hspace=0.01, wspace=0.01)
        plt.close()
        return fig
