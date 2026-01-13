import math
import SimpleITK as sitk
import numpy as np


def supersample_factors(xstep, ystep, zstep, target_xstep, target_ystep, target_zstep):
    xfactor = int(math.ceil(target_xstep / float(xstep)))
    yfactor = int(math.ceil(target_ystep / float(ystep)))
    zfactor = int(math.ceil(target_zstep / float(zstep)))
    xfactor = xfactor + 1 if xfactor % 2 == 0 else xfactor
    yfactor = yfactor + 1 if yfactor % 2 == 0 else yfactor
    zfactor = zfactor + 1 if zfactor % 2 == 0 else zfactor

    # init 0.55, 0.74, 0.74
    # target 1, 1, 3
    # upsample factors 3, 3, 5  --> 0.3, 0.3, 0.6
    #
    # target iso 1, 1, 1
    # upsample factors 3, 3, 10 -> 0.3, 0.3, 0.3
    xfactor = max(1, xfactor)
    yfactor = max(1, yfactor)
    zfactor = max(1, zfactor)

    return xfactor, yfactor, zfactor


def get_sigma(downsampling_factor, spacing):
    """Compute optimal standard deviation for Gaussian kernel.
    From Cardoso et al., "Scale factor point spread function matching:
    beyond aliasing in image resampling", MICCAI 2015
    """
    k = downsampling_factor
    variance = (k ** 2 - 1 ** 2) * (2 * np.sqrt(2 * np.log(2))) ** (-2)
    sigma = spacing * np.sqrt(variance)
    return sigma


def resample_image(
        input_image,
        like_image,
        transform,
        force_factors,
        output_spacing,
        output_size,
        upsample_method,
        regularization_method,
        downsample_method
):
    input_image = sitk.Cast(input_image, sitk.sitkFloat32)
    like_image = sitk.Cast(like_image, sitk.sitkFloat32)
    input_spacing = input_image.GetSpacing()

    if not output_spacing:
        output_spacing = like_image.GetSpacing()

    if force_factors:
        expand_factors = force_factors
    else:
        expand_factors = supersample_factors(
            input_spacing[0], input_spacing[1], input_spacing[2],
            output_spacing[0], output_spacing[1], output_spacing[2]
        )

    if not output_size:
        output_size = [
            int(math.ceil(s * lo / o))
            for s, o, lo in zip(
                like_image.GetSize(),
                output_spacing,
                like_image.GetSpacing())
        ]
    box_radius = [f // 2 for f in expand_factors]
    assert all([x % 2 == 1 and x > 0 for x in expand_factors])

    hires_like = sitk.Expand(like_image, expandFactors=expand_factors)
    hires_spacing = hires_like.GetSpacing()
    # higher smooth weight when upscaling

    if upsample_method == "nearest":
        up_interpolator = sitk.sitkNearestNeighbor
    elif upsample_method == "linear":
        up_interpolator = sitk.sitkLinear
    elif upsample_method == "bspline":
        up_interpolator = sitk.sitkBSpline3
    elif upsample_method == "sinc":
        up_interpolator = sitk.sitkLanczosWindowedSinc
    else:
        raise NotImplementedError()

    if transform:
        hires_img = sitk.Resample(
            input_image, referenceImage=hires_like,
            interpolator=up_interpolator, transform=transform)
    else:
        hires_img = sitk.Resample(
            input_image, referenceImage=hires_like, interpolator=up_interpolator)

    if regularization_method is None:
        pass
    elif regularization_method == "gaussian":
        sigma_arr = [
            get_sigma(f, h)
            for i, h, f in zip(input_spacing, hires_spacing, expand_factors)
        ]
        hires_img = sitk.SmoothingRecursiveGaussian(
            hires_img, sigma=sigma_arr, normalizeAcrossScale=False
        )
    elif regularization_method == "bilateral":
        hires_img = sitk.Bilateral(
            hires_img, domainSigma=2.0, rangeSigma=2.0
        )
    elif regularization_method == "curvature_diffusion":
        hires_img = sitk.CurvatureAnisotropicDiffusion(
            hires_img, timeStep=0.025, conductanceParameter=5.0,
            conductanceScalingUpdateInterval=1,
            numberOfIterations=50
        )
    elif regularization_method == "curvature_flow":
        hires_img = sitk.CurvatureFlow(
            hires_img, timeStep=0.03, numberOfIterations=2
        )
    elif regularization_method == "minmax_curvature_flow":
        hires_img = sitk.MinMaxCurvatureFlow(
            hires_img, timeStep=0.0625, numberOfIterations=2
        )
    else:
        raise NotImplementedError()

    if any([e != 1 for e in expand_factors]):
        if downsample_method == "mean":
            hires_img = sitk.Mean(hires_img, radius=box_radius)
        elif downsample_method == "gaussian":
            hires_img = sitk.DiscreteGaussian(
                hires_img, variance=[f // 2 for f in expand_factors],
                useImageSpacing=True
            )
        else:
            raise NotImplementedError()

    lores_img = sitk.Resample(
        size=output_size,
        image1=hires_img,
        outputOrigin=like_image.GetOrigin(),
        outputSpacing=output_spacing,
        outputDirection=like_image.GetDirection(),
        interpolator=sitk.sitkLinear,
    )

    lores_img = sitk.Cast(lores_img, sitk.sitkFloat32)
    return lores_img


def resample_label(
        input_label,
        like_image,
        transform,
        force_factors,
        output_spacing,
        output_size,
        upsample_method,
        regularization_method,
        downsample_method,
        quantile,
        absolute_threshold,
        sitk_dtype=sitk.sitkUInt8
):
    input_label = sitk.Cast(input_label, sitk.sitkUInt8)
    input_spacing = input_label.GetSpacing()
    if output_spacing:
        output_spacing = output_spacing
    else:
        output_spacing = like_image.GetSpacing()
    expand_factors = supersample_factors(
        input_spacing[0], input_spacing[1], input_spacing[2],
        output_spacing[0], output_spacing[1], output_spacing[2]
    )
    if output_size:
        output_size = output_size
    else:
        output_size = [
            int(math.ceil(s * lo / o))
            for s, o, lo in zip(
                like_image.GetSize(),
                output_spacing,
                like_image.GetSpacing())
        ]
    box_radius = [f // 2 for f in expand_factors]
    assert all([x % 2 == 1 and x > 0 for x in expand_factors])

    input_view = sitk.GetArrayViewFromImage(input_label)
    regions = set(np.unique(input_view)) - {0}

    lores_like = sitk.Cast(like_image, sitk.sitkFloat32)
    lores_like = sitk.Resample(
        size=output_size,
        image1=lores_like,
        outputOrigin=lores_like.GetOrigin(),
        outputSpacing=output_spacing,
        outputDirection=lores_like.GetDirection(),
        interpolator=sitk.sitkLinear,
    )
    hires_like = sitk.Cast(sitk.Expand(like_image, expandFactors=expand_factors), sitk.sitkFloat32)
    hires_spacing = hires_like.GetSpacing()

    tensor = []
    for i in regions:
        label_i = np.where(input_view == i, 1, 0).astype(np.float32)
        label_i_img = sitk.GetImageFromArray(label_i)
        label_i_img.CopyInformation(input_label)
        if upsample_method == "nearest":
            up_interpolator = sitk.sitkNearestNeighbor
        elif upsample_method == "linear":
            up_interpolator = sitk.sitkLinear
        elif upsample_method == "bspline":
            up_interpolator = sitk.sitkBSpline3
        elif upsample_method == "sinc":
            up_interpolator = sitk.sitkLanczosWindowedSinc
        else:
            raise NotImplementedError()

        if transform:
            hires_i_img = sitk.Resample(
                label_i_img, referenceImage=hires_like,
                interpolator=up_interpolator, transform=transform,
                outputPixelType=sitk.sitkFloat32
            )
        else:
            hires_i_img = sitk.Resample(
                label_i_img, referenceImage=hires_like,
                interpolator=up_interpolator, outputPixelType=sitk.sitkFloat32
            )

        if regularization_method is None:
            pass
        elif regularization_method == "gaussian":
            sigma_arr = [
                get_sigma(f, h)
                for i, h, f in zip(input_spacing, hires_spacing, expand_factors)
            ]
            hires_i_img = sitk.SmoothingRecursiveGaussian(
                hires_i_img, sigma=sigma_arr, normalizeAcrossScale=False
            )
        elif regularization_method == "bilateral":
            hires_i_img = sitk.Bilateral(
                hires_i_img, domainSigma=2.0, rangeSigma=2.0
            )
        elif regularization_method == "curvature_diffusion":
            hires_i_img = sitk.CurvatureAnisotropicDiffusion(
                hires_i_img, timeStep=0.025, conductanceParameter=5.0,
                conductanceScalingUpdateInterval=1,
                numberOfIterations=50
            )
        elif regularization_method == "curvature_flow":
            hires_i_img = sitk.CurvatureFlow(
                hires_i_img, timeStep=0.03, numberOfIterations=2
            )
        elif regularization_method == "minmax_curvature_flow":
            hires_i_img = sitk.MinMaxCurvatureFlow(
                hires_i_img, timeStep=0.0625, numberOfIterations=2
            )
        else:
            raise NotImplementedError()

        hires_i_img = sitk.Threshold(
            hires_i_img, lower=0.0, upper=2.0, outsideValue=0.0
        )
        if any([e != 1 for e in expand_factors]):
            if downsample_method == "mean":
                hires_i_img = sitk.Mean(hires_i_img, radius=box_radius)
            elif downsample_method == "gaussian":
                hires_i_img = sitk.DiscreteGaussian(
                    hires_i_img, variance=[f // 2 for f in expand_factors],
                    useImageSpacing=True
                )
            else:
                raise NotImplementedError()

        lores_i_img = sitk.Resample(
            hires_i_img, referenceImage=lores_like,
            interpolator=sitk.sitkLinear
        )
        tensor.append(sitk.GetArrayFromImage(lores_i_img))

    if len(tensor) >= 1:
        tensor = np.stack(tensor, axis=0)
        tensor_max = np.max(tensor, axis=0)

        tensor_argmax = np.argmax(tensor, axis=0)
        bkg_tensor_argmax = np.zeros_like(tensor_argmax)
        for i in range(tensor.shape[0]):
            tmparr = tensor[i].reshape((-1,))
            tmparr = tmparr[tmparr > 0]
            if absolute_threshold is not None:
                t = absolute_threshold
            else:
                t = np.quantile(tmparr, quantile)
            # print(" region {} using threshold {}".format(i, t))
            fg = np.where(np.logical_and(tensor_argmax == i, tensor_max >= t), 1 + i, 0)
            bkg_tensor_argmax += fg

        remap_regions = dict((r, np.where(bkg_tensor_argmax == i + 1)) for i, r in enumerate(regions))
        for r, indices in remap_regions.items():
            bkg_tensor_argmax[indices] = r
        lores_labels = bkg_tensor_argmax
    else:
        lores_labels = np.zeros_like(sitk.GetArrayFromImage(lores_like))

    lores_labels = sitk.GetImageFromArray(lores_labels)
    lores_labels.CopyInformation(lores_like)
    lores_labels = sitk.Cast(lores_labels, sitk_dtype)

    lores_view = sitk.GetArrayViewFromImage(lores_labels)

    lores_regions = set(np.unique(lores_view)) - {0}
    if regions != lores_regions:
        raise Exception("missing resampled lesion, {}".format(regions - lores_regions))

    return lores_labels
