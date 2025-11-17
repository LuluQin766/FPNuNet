
import torch
from torch import Tensor
from torchmetrics import MetricCollection, JaccardIndex, F1Score, Dice
from torchmetrics import MeanSquaredError, MeanAbsoluteError
from torchmetrics.classification import MulticlassConfusionMatrix

from torchmetrics import (
    MetricCollection, JaccardIndex, Dice, F1Score,
    MeanSquaredError, MeanAbsoluteError
)

class MulticlassDiceScore(MulticlassConfusionMatrix):
    def __init__(
        self,
        num_classes: int,
        average: str = "macro",
        ignore_index: int = None,
        validate_args: bool = True,
        **kwargs
    ):
        super().__init__(
            num_classes=num_classes,
            ignore_index=ignore_index,
            validate_args=validate_args,
            **kwargs
        )
        self.average = average

    def compute(self) -> Tensor:
        confmat = super().compute()  # shape: (C, C)
        tp = confmat.diag()
        fp = confmat.sum(dim=0) - tp
        fn = confmat.sum(dim=1) - tp

        # Dice = 2TP / (2TP + FP + FN)
        dice = 2 * tp / (2 * tp + fp + fn + 1e-7)

        if self.ignore_index is not None:
            mask = torch.ones_like(dice, dtype=torch.bool)
            mask[self.ignore_index] = False
            dice = dice[mask]

        if self.average == "macro":
            return dice.mean()
        elif self.average in ("none", None):
            return dice
        else:
            raise ValueError(f"Unsupported average mode: {self.average}")

def get_binary_metrics(ignore_index: int = None):
    return MetricCollection({
        "iou_micro": JaccardIndex(task="binary", ignore_index=ignore_index),
        "dice_macro": Dice(
                        num_classes=2,
                        threshold=0.5,
                        average="macro",          # 建议 macro
                        mdmc_average="global",    # 适用于 [B,1,H,W]
                        ignore_index=ignore_index,
                        zero_division=0
                    ),
        "f1_macro": F1Score(task="binary", average="macro", ignore_index=ignore_index),
    })

def get_multiclass_metrics(num_classes: int, ignore_index: int = 0):
    return MetricCollection({
        "dice_macro": Dice(num_classes=num_classes, average='macro', ignore_index=ignore_index),
        "iou_macro": JaccardIndex(task="multiclass", num_classes=num_classes, average='macro', ignore_index=ignore_index),
        "f1_macro": F1Score(task="multiclass", num_classes=num_classes, average="macro", ignore_index=ignore_index),
        # Class-wise metrics
        "dice_classwise": MulticlassDiceScore(num_classes=num_classes, ignore_index=ignore_index, average=None),
        "iou_classwise": JaccardIndex(task="multiclass", num_classes=num_classes, average=None, ignore_index=ignore_index),
    })

def get_regression_metrics():
    return MetricCollection({
        "mse": MeanSquaredError(),
        "mae": MeanAbsoluteError(),
    })

def get_metrics(num_classes: int, ignore_index: int = 0, output_aux: bool = False):
    """
    Returns a dict of MetricCollections for:
      - Main and auxiliary outputs of bin, boundary, tp (type), and hv branches.
    """
    metrics = {
        # === Main branches ===
        "bin": get_binary_metrics(),
        "boundary": get_binary_metrics(),
        "tp": get_multiclass_metrics(num_classes=num_classes, ignore_index=ignore_index),
        "hv": get_regression_metrics(),
    }

    if output_aux:
        metrics.update({
            # === Auxiliary branches ===
            "bin_aux_0": get_binary_metrics(),
            "bin_aux_1": get_binary_metrics(),

            "type_aux_0": get_multiclass_metrics(num_classes=num_classes, ignore_index=ignore_index),
            "type_aux_1": get_multiclass_metrics(num_classes=num_classes, ignore_index=ignore_index),
            "type_aux_2": get_multiclass_metrics(num_classes=num_classes, ignore_index=ignore_index),

            "hv_aux_0": get_regression_metrics(),
            "hv_aux_1": get_regression_metrics(),
        })

    return metrics
