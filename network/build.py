"""Model construction and checkpoint loading."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .fpnunet import FPNuNet


def _unwrap_state_dict(state: Any) -> dict[str, torch.Tensor]:
    if not isinstance(state, dict):
        raise TypeError("Checkpoint must contain a mapping of parameter names to tensors")
    for container_key in ("state_dict", "model_state_dict", "model"):
        candidate = state.get(container_key)
        if isinstance(candidate, dict):
            state = candidate
            break
    return {str(key): value for key, value in state.items() if torch.is_tensor(value)}


def _strip_training_prefixes(key: str) -> str:
    prefixes = ("module.", "model.", "network.", "fpnunet.")
    changed = True
    while changed:
        changed = False
        for prefix in prefixes:
            if key.startswith(prefix):
                key = key[len(prefix) :]
                changed = True
    return key


def _load_uni(checkpoint: str | Path):
    try:
        import timm
    except ImportError as error:
        raise ImportError("Install timm to construct the UNI ViT-L backbone") from error
    model = timm.create_model(
        "vit_large_patch16_224",
        img_size=256,
        patch_size=16,
        num_classes=0,
        dynamic_img_size=False,
    )
    model.patch_embed.strict_img_size = False
    state = _unwrap_state_dict(
        torch.load(checkpoint, map_location="cpu", weights_only=False)
    )
    current = model.state_dict()
    compatible = {
        _strip_training_prefixes(key): value
        for key, value in state.items()
        if _strip_training_prefixes(key) in current
        and current[_strip_training_prefixes(key)].shape == value.shape
    }
    matched_parameters = sum(value.numel() for value in compatible.values())
    expected_parameters = sum(value.numel() for value in current.values())
    if matched_parameters < 0.9 * expected_parameters:
        raise RuntimeError(
            "UNI checkpoint is incompatible with vit_large_patch16_224: "
            f"matched {matched_parameters:,}/{expected_parameters:,} parameters"
        )
    model.load_state_dict(compatible, strict=False)
    return model


def _load_sam(checkpoint: str | Path):
    try:
        from segment_anything import sam_model_registry
    except ImportError:
        try:
            from segment_anything_local import sam_model_registry
        except ImportError as error:
            raise ImportError(
                "Install Meta's segment-anything package to construct SAM ViT-B"
            ) from error
    sam = sam_model_registry["vit_b"](checkpoint=str(checkpoint))
    return sam.image_encoder


def build_model(
    sam_checkpoint: str | Path,
    uni_checkpoint: str | Path,
    fpnunet_checkpoint: str | Path | None = None,
    device: str | torch.device = "cpu",
) -> FPNuNet:
    """Build the paper model and optionally load a trained FPNuNet state dict."""
    for name, value in (("SAM", sam_checkpoint), ("UNI", uni_checkpoint)):
        if not Path(value).is_file():
            raise FileNotFoundError(f"{name} checkpoint not found: {value}")
    model = FPNuNet(_load_sam(sam_checkpoint), _load_uni(uni_checkpoint))
    if fpnunet_checkpoint is not None:
        state = _unwrap_state_dict(
            torch.load(fpnunet_checkpoint, map_location="cpu", weights_only=False)
        )
        state = {_strip_training_prefixes(key): value for key, value in state.items()}
        model.load_state_dict(state, strict=True)
    return model.to(device)
