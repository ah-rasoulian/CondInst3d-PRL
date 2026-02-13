from condinst3d.io.datamodule.nndet_result import nnDetectionResult
from condinst3d.io.datamodule.nnunet_result import nnUNetResult


def main(
        split_root = "/scratch/04/public_datasets/nnDet/Task100_PRL/raw_splitted",
        n_modalities = 4,
        pred_root = "/scratch/04/public_datasets/nnUNet_predictions"
):
    # dm = nnUNetResult(split_root, pred_root, n_modalities)
    dm = nnDetectionResult(split_root, pred_root, n_modalities, score_threshold=0.5)
    dm.setup("test")
    for batch in dm.test_dataloader():
        print(batch)
        # break

if __name__ == "__main__":
    main(
        pred_root="/scratch/01/ahrasoulian/projects/nnDetection/models/Task100_PRL/RetinaUNetV001_D3V001_3d/fold0/test_predictions"
    )