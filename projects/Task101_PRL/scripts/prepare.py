from __future__ import annotations
from concurrent.futures import ProcessPoolExecutor, as_completed
import shutil
import os
import sys
from pathlib import Path
import subprocess
import numpy as np
import pandas as pd
from loguru import logger
import json
from tqdm import tqdm
import SimpleITK as sitk
from pathlib import Path
import json
from concurrent.futures import ProcessPoolExecutor, as_completed

from condinst3d.io import save_json
from condinst3d.io.load import sitk_read_image, sitk_read_transform, sitk_write_image, sitk_read_label
from condinst3d.io.prepare import create_random_test_split, create_stratified_test_split, create_splits_final_pkl
from condinst3d.utils.check import env_guard, check_spacing
from condinst3d.utils.info import maybe_verbose_iterable
from condinst3d.io.resample import resample_image, resample_label


def _convert2nifty_one_case(task):
    case_id = task["case_id"]
    case_dir = Path(task["case_dir"])
    source_data_dir = task["source_data_dir"]
    mods_xfms = task["mods_xfms"]
    visit = task["visit"]

    visit_timepoint = visit["timepoint"]
    visit_dir = case_dir / visit_timepoint
    os.makedirs(visit_dir, exist_ok=True)

    # write prl file
    cprl_swi_path = source_data_dir + visit["cprl_swi"]
    assert check_spacing(cprl_swi_path, [0.8, 0.8, 2.0]), f"Bad spacing for {cprl_swi_path}"

    cprl_swi = sitk_read_label(str(cprl_swi_path))
    cprl_swi_array = sitk.GetArrayFromImage(cprl_swi)

    instances = list(np.unique(cprl_swi_array))
    if 0 in instances:
        instances.remove(0)

    instances_dict = {}
    cprl_swi_array_sequential = np.zeros_like(cprl_swi_array)

    for new_idx, inst_idx in enumerate(instances, start=1):
        cprl_swi_array_sequential[cprl_swi_array == inst_idx] = new_idx
        instances_dict[str(new_idx)] = 0

    cprl_swi_sequential = sitk.GetImageFromArray(cprl_swi_array_sequential)
    cprl_swi_sequential.CopyInformation(cprl_swi)

    sitk_write_image(cprl_swi_sequential, str(visit_dir / "prl.nii.gz"))
    save_json({"instances": instances_dict}, str(visit_dir / "prl.json"))

    # write t2lesion file
    ct2f_stx_path = source_data_dir + visit["ct2f_stx"]
    ct2f_stx = sitk_read_label(str(ct2f_stx_path))

    r_label = resample_label(
        input_label=ct2f_stx,
        like_image=cprl_swi_sequential,
        transform=None,
        output_spacing=cprl_swi_sequential.GetSpacing(),
        output_size=cprl_swi_sequential.GetSize(),
        upsample_method='linear',
        downsample_method='mean',
        quantile=None,
        absolute_threshold=0.5,
        force_factors=None,
        regularization_method=None
    )
    sitk_write_image(r_label, str(visit_dir / "t2lesion.nii.gz"))

    # write modalities
    for img_key, xfm_key in mods_xfms:
        img_path = source_data_dir + visit[img_key]
        xfm_path = source_data_dir + visit[xfm_key]

        img = sitk_read_image(str(img_path))
        xfm = sitk_read_transform(str(xfm_path))

        r_img = resample_image(
            input_image=img,
            like_image=cprl_swi_sequential,
            transform=xfm,
            output_spacing=cprl_swi_sequential.GetSpacing(),
            output_size=cprl_swi_sequential.GetSize(),
            upsample_method='bspline',
            downsample_method='mean',
            force_factors=None,
            regularization_method=None
        )
        sitk_write_image(r_img, str(visit_dir / f"{img_key}.nii.gz"))

    return {
        "case_id": case_id,
        "info": task["info_value"],
    }


def convert_to_nifty(num_workers=None):
    t_order = [
        "screening", "LTS-UV1", "LTS-baseline", "LTS-m03", "w24", "m06", "w48",
        "LTS-m12", "m12", "w72", "m18", "w96", "m24", "m36", "m48", "EOT",
        "OLE-EOT", "pEOT", "LTS-pEOT", "EOS"
    ]
    t_order_map = {tp: i for i, tp in enumerate(t_order)}

    dataset_csvs = [
        "/scratch/01/ahrasoulian/datasets/prlnet/2025-12-02/prl_dataset_wide_2025.csv.gz",
        "/scratch/01/ahrasoulian/datasets/prlnet/2025-12-02/prl_dataset_wide_2024.csv.gz",
    ]

    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / "Task101_PRL"
    target_data_dir = task_data_dir / "raw"
    source_data_dir = "/scratch/01/ahrasoulian/datasets/prlnet"

    mods_xfms = [
        ("t1w", "t1w_xfm"),
        ("t2w", "t2w_xfm"),
        ("flr", "flr_xfm"),
        ("freqmap", "freqmap_xfm"),
    ]

    os.makedirs(target_data_dir, exist_ok=True)

    ds = pd.concat((pd.read_csv(f) for f in dataset_csvs), ignore_index=True)

    # rank visits
    ds["t_order_idx"] = ds["timepoint"].map(t_order_map)

    unknown = ds.loc[ds["t_order_idx"].isna(), "timepoint"].dropna().unique()
    if len(unknown) > 0:
        raise ValueError(f"Unknown timepoints found: {sorted(unknown.tolist())}")

    # keep only last visit per case
    last_visit_idx = ds.groupby(["trial", "site", "subject"], sort=False)["t_order_idx"].idxmax()
    ds = ds.loc[last_visit_idx].reset_index(drop=True)

    # build tasks
    tasks = []
    for case_number, visit in enumerate(ds.itertuples(index=False)):
        case_id = f"case_{case_number:05d}"
        case_dir = target_data_dir / case_id

        visit_dict = visit._asdict()
        info_value = f"{visit_dict['trial']}_{visit_dict['site']}_{visit_dict['subject']}"

        tasks.append({
            "case_id": case_id,
            "case_dir": str(case_dir),
            "source_data_dir": source_data_dir,
            "mods_xfms": mods_xfms,
            "visit": visit_dict,
            "info_value": info_value,
        })

    if num_workers is None:
        num_workers = min(os.cpu_count() or 1, len(tasks))

    info = {}
    failures = []

    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futures = [ex.submit(_convert2nifty_one_case, task) for task in tasks]

        for fut in tqdm(as_completed(futures), total=len(futures)):
            try:
                result = fut.result()
                info[result["case_id"]] = result["info"]
            except Exception as e:
                failures.append(str(e))

    if failures:
        raise RuntimeError(
            f"{len(failures)} case(s) failed.\nFirst few errors:\n" +
            "\n".join(failures[:10])
        )

    info_path = target_data_dir / "dataset_info.json"
    save_json(info, info_path)

    print(f"[INFO] Saved dataset info to {info_path}")


@env_guard
def split_data():
    det_data_dir = Path(os.getenv('det_data'))
    task_data_dir = det_data_dir / "Task101_PRL"
    source_data_dir = task_data_dir / "raw"

    if not source_data_dir.is_dir():
        raise RuntimeError(f"{source_data_dir} should contain the raw data but does not exist.")

    splitted_dir = task_data_dir / "raw_splitted"
    target_data_dir = task_data_dir / "raw_splitted" / "imagesTr"
    target_data_dir.mkdir(exist_ok=True, parents=True)
    target_label_dir = task_data_dir / "raw_splitted" / "labelsTr"
    target_label_dir.mkdir(exist_ok=True, parents=True)

    logger.remove()
    logger.add(sys.stdout, level="INFO")
    logger.add(task_data_dir / "prepare.log", level="DEBUG")

    dataset_info = {
        "task": "Task101_PRL",

        "name": "PRL",
        "dim": 3,

        "target_class": 0,
        "test_labels": False,

        "labels": {
            "0": "prl",
        },

        "modalities": {
            "0": "t1w",
            "1": "t2w",
            "2": "flr",
            "3": "freqmap",
        },
    }

    save_json(dataset_info, task_data_dir / "dataset.json")

    # prepare cases
    cases = [str(c.name) for c in source_data_dir.iterdir() if c.is_dir()]
    for c in maybe_verbose_iterable(cases):
        logger.info(f"Copy case {c}")
        visits = [str(v.name) for v in (source_data_dir / c).iterdir() if v.is_dir()]

        # copying only one visit per subject for simplicity
        shutil.copy(source_data_dir / c / visits[0] / "t1w.nii.gz", target_data_dir / f"{c}_0000.nii.gz")
        shutil.copy(source_data_dir / c / visits[0] / "t2w.nii.gz", target_data_dir / f"{c}_0001.nii.gz")
        shutil.copy(source_data_dir / c / visits[0] / "flr.nii.gz", target_data_dir / f"{c}_0002.nii.gz")
        shutil.copy(source_data_dir / c / visits[0] / "freqmap.nii.gz", target_data_dir / f"{c}_0003.nii.gz")
        shutil.copy(source_data_dir / c / visits[0] / "prl.nii.gz", target_label_dir / f"{c}.nii.gz")
        shutil.copy(source_data_dir / c / visits[0] / "prl.json", target_label_dir / f"{c}.json")

    # create an artificial test split
    create_stratified_test_split(splitted_dir=splitted_dir,
                                 num_modalities=4,
                                 test_size=0.2,
                                 random_state=0,
                                 shuffle=True,
                                 )

def _brain_extract_one_case(case_t1: str, out_mask: str, sif: str, bind_args: list[str]) -> tuple[str, bool, str]:
    """
    Runs synthstrip for one file. Returns (case_t1, ok, message).
    """
    case_t1_p = Path(case_t1)
    out_mask_p = Path(out_mask)

    if out_mask_p.exists():
        return (case_t1, True, "skipped")

    cmd = [
        "singularity", "exec",
        *bind_args,
        sif,
        "mri_synthstrip",
        "-i", str(case_t1_p),
        "-m", str(out_mask_p),
        "--no-csf",
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        return (case_t1, True, "ok")
    except subprocess.CalledProcessError as e:
        msg = (
            f"Command: {' '.join(cmd)}\n"
            f"STDOUT:\n{e.stdout}\n\n"
            f"STDERR:\n{e.stderr}\n"
        )
        return (case_t1, False, msg)


def brain_extract(num_workers: int = 8, fail_fast: bool = False) -> None:
    det_data_dir = Path(os.getenv("det_data"))
    task_data_dir = det_data_dir / "Task101_PRL"
    splitted_dir = task_data_dir / "raw_splitted"

    images_tr = splitted_dir / "imagesTr"
    images_ts = splitted_dir / "imagesTs"

    sif = Path(os.getenv("FREESURFER_SIF", str(Path.home() / "containers" / "freesurfer_7.4.1.sif")))
    if not sif.exists():
        raise FileNotFoundError(
            f"FreeSurfer SIF not found: {sif}\n"
            f"Set FREESURFER_SIF or place it at ~/containers/freesurfer_7.4.1.sif"
        )

    bind_roots = ["/scratch", "/trials", str(Path.home())]
    bind_args: list[str] = []
    for b in bind_roots:
        if Path(b).exists():
            bind_args += ["-B", f"{b}:{b}"]

    # Collect jobs
    jobs: list[tuple[str, str]] = []
    for data_dir in [images_tr, images_ts]:
        if not data_dir.exists():
            continue
        brainmask_dir = data_dir / "brainmask"
        os.makedirs(brainmask_dir, exist_ok=True)
        for case_t1 in sorted(data_dir.glob("*_0000.nii.gz")):
            case_name = case_t1.name.removesuffix("_0000.nii.gz")
            out_mask = brainmask_dir / f"{case_name}_brainmask.nii.gz"
            if out_mask.exists():
                continue
            jobs.append((str(case_t1), str(out_mask)))

    if not jobs:
        print("No missing masks found. Nothing to do.")
        return

    print(f"Running SynthStrip on {len(jobs)} cases with {num_workers} workers...")

    errors: list[str] = []
    with ProcessPoolExecutor(max_workers=num_workers) as ex:
        futs = [
            ex.submit(_brain_extract_one_case, case_t1, out_mask, str(sif), bind_args)
            for case_t1, out_mask in jobs
        ]

        for fut in tqdm(as_completed(futs), total=len(futs)):
            case_t1, ok, msg = fut.result()
            if not ok:
                errors.append(f"FAILED: {case_t1}\n{msg}")
                if fail_fast:
                    raise RuntimeError(errors[-1])

    if errors:
        # Write a log so you can re-run only failures
        log_path = splitted_dir / "synthstrip_failures.log"
        log_path.write_text("\n\n" + ("\n" + "-" * 80 + "\n").join(errors))
        raise RuntimeError(f"{len(errors)} SynthStrip jobs failed. See: {log_path}")

    print("All SynthStrip jobs completed successfully.")


def create_splits_final():
    splits = create_splits_final_pkl(
        splitted_dir="/scratch/04/public_datasets/nnDet/Task101_PRL/raw_splitted",
        out_path="/scratch/04/public_datasets/nnDet/Task101_PRL/preprocessed/splits_final.pkl",
        n_splits=5,
        random_state=0,
    )
    print(splits)


if __name__ == '__main__':
    from fire import Fire
    Fire({
        "convert2nifty": convert_to_nifty,
        "split_data": split_data,
        "brain_extract": brain_extract,
        "create_splits": create_splits_final,
    })
