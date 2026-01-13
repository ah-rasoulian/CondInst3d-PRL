import functools
import os
import warnings
import numpy as np
import subprocess
from condinst3d.io.paths import find_minctools_path


def env_guard(func):
    """
    Contextmanager to check nnDetection environment variables
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        # we use print here because logging might not be initialized yet and
        # this is intended as a user warning.

        # det_data
        if os.environ.get("det_data", None) is None:
            raise RuntimeError(
                "'det_data' environment variable not set. "
                "Please refer to the installation instructions. "
            )

        # det_models
        if os.environ.get("det_models", None) is None:
            raise RuntimeError(
                "'det_models' environment variable not set. "
                "Please refer to the installation instructions. "
            )

        # OMP_NUM_THREADS
        if os.environ.get("OMP_NUM_THREADS", None) is None:
            raise RuntimeError(
                "'OMP_NUM_THREADS' environment variable not set. "
                "Please refer to the installation instructions. "
            )

        # det_num_threads
        if os.environ.get("det_num_threads", None) is None:
            warnings.warn(
                "Warning: 'det_num_threads' environment variable not set. "
                "Please read installation instructions again. "
                "Training will not work properly.")

        # det_verbose
        if os.environ.get("det_verbose", None) is None:
            print("'det_verbose' environment variable not set. "
                  "Continue in verbose mode.")

        return func(*args, **kwargs)

    return wrapper


def get_minc_spacing(minc_path):
    minc_bin = find_minctools_path(binary='mincinfo')
    # Get the spacing along x, y, and z dimensions
    x_output = subprocess.check_output([minc_bin, "-attvalue", "xspace:step", minc_path])
    y_output = subprocess.check_output([minc_bin, "-attvalue", "yspace:step", minc_path])
    z_output = subprocess.check_output([minc_bin, "-attvalue", "zspace:step", minc_path])

    # Convert from bytes and strip whitespace, then to float
    x_spacing = float(x_output.decode("utf-8").strip())
    y_spacing = float(y_output.decode("utf-8").strip())
    z_spacing = float(z_output.decode("utf-8").strip())

    spacing = np.array([x_spacing, y_spacing, z_spacing])
    return spacing


def check_spacing(path, sp):
    spacing = get_minc_spacing(path)
    if (np.abs(spacing - np.array(sp))).sum() < 0.1:
        return True
    else:
        return False
