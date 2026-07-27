"""Paper-specified multi-task training objective."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def _dice_binary(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probability = logits.sigmoid()
    intersection = (probability * target).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + target.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def _focal_binary(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0):
    bce = F.binary_cross_entropy_with_logits(logits, target, reduction="none")
    probability = logits.sigmoid()
    pt = probability * target + (1.0 - probability) * (1.0 - target)
    return ((1.0 - pt).pow(gamma) * bce).mean()


def _one_hot(target: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(target.long(), classes).permute(0, 3, 1, 2).float()


def _type_dice(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probability = logits.softmax(dim=1)[:, 1:]
    truth = _one_hot(target, logits.shape[1])[:, 1:]
    intersection = (probability * truth).sum(dim=(-2, -1))
    denominator = probability.sum(dim=(-2, -1)) + truth.sum(dim=(-2, -1))
    return 1.0 - ((2.0 * intersection + eps) / (denominator + eps)).mean()


def _type_focal(logits: torch.Tensor, target: torch.Tensor, gamma: float = 2.0):
    ce = F.cross_entropy(logits, target.long(), reduction="none")
    pt = logits.softmax(1).gather(1, target.long().unsqueeze(1)).squeeze(1)
    foreground = target > 0
    if foreground.any():
        return ((1.0 - pt[foreground]).pow(gamma) * ce[foreground]).mean()
    return ce.mean() * 0.0


def _soft_iou(logits: torch.Tensor, target: torch.Tensor, eps: float = 1e-6):
    probability = logits.softmax(dim=1)[:, 1:]
    truth = _one_hot(target, logits.shape[1])[:, 1:]
    intersection = (probability * truth).sum(dim=(-2, -1))
    union = (probability + truth - probability * truth).sum(dim=(-2, -1))
    return 1.0 - ((intersection + eps) / (union + eps)).mean()


def _sobel_gradients(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    kx = x.new_tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).reshape(1, 1, 3, 3)
    ky = kx.transpose(-1, -2)
    channels = x.shape[1]
    return (
        F.conv2d(x, kx.repeat(channels, 1, 1, 1), padding=1, groups=channels),
        F.conv2d(x, ky.repeat(channels, 1, 1, 1), padding=1, groups=channels),
    )


class FPNuNetLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.register_buffer(
            "type_weights", torch.tensor([0.25, 1.0, 1.0, 1.0, 1.0])
        )

    def forward(
        self,
        prediction: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        binary_target = target["bin"].float()
        boundary_target = target["boundary"].float()
        binary_core = (
            F.binary_cross_entropy_with_logits(prediction["bin"], binary_target)
            + _dice_binary(prediction["bin"], binary_target)
            + _focal_binary(prediction["bin"], binary_target)
        )
        boundary = (
            F.binary_cross_entropy_with_logits(prediction["boundary"], boundary_target)
            + _dice_binary(prediction["boundary"], boundary_target)
            + _focal_binary(prediction["boundary"], boundary_target)
        )
        binary_loss = binary_core + 0.5 * boundary

        foreground = binary_target.expand_as(prediction["hv"])
        normalizer = foreground.sum().clamp_min(1.0)
        hv_mse = ((prediction["hv"] - target["hv"]).pow(2) * foreground).sum() / normalizer
        pred_gx, pred_gy = _sobel_gradients(prediction["hv"])
        true_gx, true_gy = _sobel_gradients(target["hv"])
        hv_msge = (
            ((pred_gx - true_gx).pow(2) + (pred_gy - true_gy).pow(2)) * foreground
        ).sum() / normalizer
        hv_loss = hv_mse + hv_msge

        type_target = target["tp"].long()
        type_loss = (
            F.cross_entropy(prediction["tp"], type_target, weight=self.type_weights)
            + _type_dice(prediction["tp"], type_target)
            + _type_focal(prediction["tp"], type_target)
            + _soft_iou(prediction["tp"], type_target)
        )
        total = binary_loss + hv_loss + type_loss
        return {"total": total, "binary": binary_loss, "hv": hv_loss, "type": type_loss}
