"""HoVer-Net-style watershed post-processing."""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage as ndi
from skimage import morphology, segmentation


def logits_to_instances(
    binary_logits: torch.Tensor,
    hv_map: torch.Tensor,
    type_logits: torch.Tensor,
    threshold: float = 0.5,
    min_size: int = 10,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert one model output into an instance map and majority type map."""
    binary = binary_logits.detach().float().sigmoid().squeeze().cpu().numpy() > threshold
    binary = morphology.remove_small_objects(binary, min_size=min_size)
    if not binary.any():
        shape = binary.shape
        return np.zeros(shape, np.int32), np.zeros(shape, np.uint8)
    hv = hv_map.detach().float().cpu().numpy()
    horizontal = ndi.sobel(hv[0], axis=1)
    vertical = ndi.sobel(hv[1], axis=0)
    energy = np.maximum(np.abs(horizontal), np.abs(vertical))
    distance = ndi.distance_transform_edt(binary)
    seeds = morphology.local_maxima(distance - energy)
    markers, _ = ndi.label(seeds & binary)
    instances = segmentation.watershed(-distance + energy, markers, mask=binary).astype(np.int32)
    pixel_types = type_logits.detach().float().argmax(0).cpu().numpy()
    instance_types = np.zeros_like(pixel_types, dtype=np.uint8)
    for instance_id in np.unique(instances):
        if instance_id == 0:
            continue
        mask = instances == instance_id
        labels = pixel_types[mask]
        foreground = labels[labels > 0]
        chosen = np.bincount(foreground if foreground.size else labels).argmax()
        instance_types[mask] = chosen
    return instances, instance_types
