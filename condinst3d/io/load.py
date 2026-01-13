from typing import Any
import json
from condinst3d.io.paths import Pathlike, Path
from pathlib import Path
import SimpleITK as sitk
import tempfile
import shutil
import subprocess
import os
from contextlib import contextmanager
from condinst3d.io.paths import find_minctools_path


MINC2_ENV = {
            "MINC_COMPRESS": "4",
            "MINC_FORCE_V2": "1",
        }
MINC1_ENV = {
            "MINC_COMPRESS": "4",
            "MINC_FORCE_V2": "0",
        }


def save_json(data: Any, path: Pathlike, indent: int = 4, **kwargs):
    """
    Load json file

    Args:
        data: data to save to json
        path: path to json file
        indent: passed to json.dump
        **kwargs: keyword arguments passed to :func:`json.dump`
    """
    if isinstance(path, str):
        path = Path(path)
    if not (".json" == path.suffix):
        path = Path(str(path) + ".json")

    with open(path, "w") as f:
        json.dump(data, f, indent=indent, **kwargs)


def sitk_read_transform(filepath):
    if filepath is None:
        transform = None
    else:
        transform = sitk.ReadTransform(filepath)
        if filepath.endswith(".xfm"):
            transform = transform.GetInverse()
    return transform


def sitk_read_label(filepath, tmpdir=None, format="minc2"):
    with maybe_convert_to(filepath, tmpdir, format) as f:
        return sitk.Cast(sitk.Round(sitk.ReadImage(f, sitk.sitkFloat32)), sitk.sitkUInt8)


def sitk_read_image(filepath, tmpdir=None, format="minc2"):
    with maybe_convert_to(filepath, tmpdir, format) as f:
        img = sitk.ReadImage(f, sitk.sitkFloat32)
        return img


def sitk_write_image(image, filepath):
    if filepath.endswith(".mnc.gz"):
        mincresample = find_minctools_path()
        tmpdir = tempfile.mkdtemp()
        try:
            tmppath1 = os.path.join(tmpdir, "tmp.1.mnc")
            tmppath2 = os.path.join(tmpdir, "tmp.2.mnc")
            sitk.WriteImage(image, tmppath2, useCompression=True)
            subprocess.check_call([
                mincresample, "-quiet", tmppath2, tmppath1
            ], stdout=subprocess.DEVNULL, env=MINC1_ENV)
            subprocess.check_call(["gzip", "-f", tmppath1])
            shutil.copyfile(tmppath1 + ".gz", filepath)
        finally:
            shutil.rmtree(tmpdir)
    else:
        sitk.WriteImage(image, filepath, useCompression=True)
    return


@contextmanager
def maybe_convert_to(filepath, dir=None, format="minc2"):
    if format == "minc2":
        with maybe_convert_to_minc2(filepath, dir) as f:
            yield f
    elif format == "nifti":
        with maybe_convert_to_nifti(filepath, dir) as f:
            yield f
    else:
        raise ValueError("unknown format {}".format(format))


@contextmanager
def maybe_convert_to_minc2(filepath, dir=None):
    if filepath.endswith(".mnc.gz"):
        mincresample = find_minctools_path()
        if dir is not None:
            os.makedirs(dir, exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=dir)
        try:
            tmppath = os.path.join(tmpdir, "tmp.mnc")
            subprocess.check_call([
                mincresample, "-2", "-quiet", filepath, tmppath
            ], stdout=subprocess.DEVNULL, env=MINC2_ENV)
            yield tmppath
        finally:
            shutil.rmtree(tmpdir)
    else:
        yield filepath


@contextmanager
def maybe_convert_to_nifti(filepath, dir=None):
    if filepath.endswith(".mnc.gz") or filepath.endswith(".mnc"):
        mnc2nii = find_minctools_path(binary="mnc2nii")
        if dir is not None:
            os.makedirs(dir, exist_ok=True)
        tmpdir = tempfile.mkdtemp(dir=dir)
        try:
            tmppath = os.path.join(tmpdir, "tmp.nii")
            subprocess.check_call([
                mnc2nii, filepath, tmppath
            ], stdout=subprocess.DEVNULL, env=MINC2_ENV)
            yield tmppath
        finally:
            shutil.rmtree(tmpdir)
    else:
        yield filepath
