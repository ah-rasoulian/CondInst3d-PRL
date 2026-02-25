from typing import Optional, Iterable
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from condinst3d.io.datamodule.nndet_result import nnDetectionResult
from condinst3d.io.datamodule.nnunet_result import nnUNetResult
from condinst3d.io.datamodule.condinst_result import CondInstResult
from condinst3d.arch.model_evaluator import ModelEvaluator


def setup_logger(
    save_dir: str = "logs",
    name: str = "tb",
    version: Optional[str] = None,
    iou_thresholds: Optional[Iterable[float]] = None,
    add_custom_scalars: bool = True,
) -> TensorBoardLogger:
    """
    Create a default TensorBoardLogger with optional custom scalar layouts.

    Args:
        save_dir: Root directory for logs
        name: Experiment name
        version: Logger version (None = auto)
        iou_thresholds: IoU thresholds for custom scalar layout
                        (e.g. [0.5, 0.4, 0.3])
        add_custom_scalars: Whether to register TensorBoard custom scalars

    Returns:
        TensorBoardLogger
    """

    logger = TensorBoardLogger(
        save_dir=save_dir,
        name=name,
        version=version,
        default_hp_metric=False,
    )

    if add_custom_scalars and iou_thresholds:
        iou_list = [float(x) for x in iou_thresholds]

        layout = {
            "Masks-IoU": {
                "TP": ["Multiline", [f"Masks-IoU/TP@{th:.2f}" for th in iou_list]],
                "FP": ["Multiline", [f"Masks-IoU/FP@{th:.2f}" for th in iou_list]],
                "FN": ["Multiline", [f"Masks-IoU/FN@{th:.2f}" for th in iou_list]],
                "Precision": ["Multiline", [f"Masks-IoU/Precision@{th:.2f}" for th in iou_list]],
                "Recall": ["Multiline", [f"Masks-IoU/Recall@{th:.2f}" for th in iou_list]],
                "F1": ["Multiline", [f"Masks-IoU/F1@{th:.2f}" for th in iou_list]],
                "AP": ["Multiline", [f"Masks-IoU/AP@{th:.2f}" for th in iou_list]],
            },
            "Boxes-IoU": {
                "TP": ["Multiline", [f"Boxes-IoU/TP@{th:.2f}" for th in iou_list]],
                "FP": ["Multiline", [f"Boxes-IoU/FP@{th:.2f}" for th in iou_list]],
                "FN": ["Multiline", [f"Boxes-IoU/FN@{th:.2f}" for th in iou_list]],
                "Precision": ["Multiline", [f"Boxes-IoU/Precision@{th:.2f}" for th in iou_list]],
                "Recall": ["Multiline", [f"Boxes-IoU/Recall@{th:.2f}" for th in iou_list]],
                "F1": ["Multiline", [f"Boxes-IoU/F1@{th:.2f}" for th in iou_list]],
                "AP": ["Multiline", [f"Boxes-IoU/AP@{th:.2f}" for th in iou_list]],
            },
        }

        logger.experiment.add_custom_scalars(layout)

    return logger


def main(
    pred_root,
    split_root = "/scratch/04/public_datasets/nnDet/Task100_PRL/raw_splitted",
    n_modalities = 4,
):
    # dm = nnUNetResult(split_root, pred_root, n_modalities, n_workers=32)
    # dm = nnDetectionResult(split_root, pred_root, n_modalities, n_workers=32, score_threshold=0.1, iou_threshold=0.1)
    dm = CondInstResult(split_root, pred_root, n_modalities, n_workers=32)

    model = ModelEvaluator(target_key="targets", pred_key="preds")
    logger = setup_logger(name="CondInst3D")
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=[0],
        logger=logger,

        num_sanity_val_steps=0,
        enable_checkpointing=False,
        enable_progress_bar=True,
        log_every_n_steps=1,

        inference_mode=True,
        deterministic=True,
    )
    trainer.test(model, datamodule=dm)

if __name__ == "__main__":
    nndet_pred_root = "/scratch/01/ahrasoulian/projects/nnDetection/models/Task100_PRL/RetinaUNetV001_D3V001_3d/fold0/test_predictions"
    nnunet_pred_root = "/scratch/04/public_datasets/nnUNet_predictions"
    condinst_pred_root = "/scratch/01/ahrasoulian/projects/CondInst3d-PRL/scripts/outputs/condinst3d-prl-final/2026-02-21_00-58-25/tb/version_0/test_outputs"

    main(condinst_pred_root)
