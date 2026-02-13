from typing import Any

import pytorch_lightning as pl
from pytorch_lightning.utilities.types import STEP_OUTPUT


class BaselineEvaluator(pl.LightningModule):
    def __init__(self):
        super().__init__()

    def on_validation_start(self) -> None:
        pass

    def validation_step(self, batch: dict, batch_idx: int) -> STEP_OUTPUT:
        pass

    def on_validation_end(self) -> None:
        pass
