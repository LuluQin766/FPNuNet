"""Differentiable, real-valued orthonormal DCT utilities."""

from __future__ import annotations

import math
from functools import lru_cache

import torch


@lru_cache(maxsize=32)
def _cpu_dct_matrix(size: int) -> torch.Tensor:
    n = torch.arange(size, dtype=torch.float64)
    k = torch.arange(size, dtype=torch.float64).unsqueeze(1)
    matrix = torch.cos(math.pi * (n + 0.5) * k / size)
    matrix[0] *= math.sqrt(1.0 / size)
    matrix[1:] *= math.sqrt(2.0 / size)
    return matrix


def _dct_matrix(size: int, reference: torch.Tensor) -> torch.Tensor:
    return _cpu_dct_matrix(size).to(device=reference.device, dtype=reference.dtype)


def dct_2d(x: torch.Tensor) -> torch.Tensor:
    """Apply an orthonormal DCT-II to the last two dimensions."""
    if x.ndim < 2:
        raise ValueError("dct_2d expects at least two dimensions")
    h, w = x.shape[-2:]
    ch = _dct_matrix(h, x)
    cw = _dct_matrix(w, x)
    return torch.matmul(torch.matmul(ch, x), cw.transpose(0, 1))


def idct_2d(x: torch.Tensor) -> torch.Tensor:
    """Invert :func:`dct_2d` using the transpose of its orthonormal basis."""
    if x.ndim < 2:
        raise ValueError("idct_2d expects at least two dimensions")
    h, w = x.shape[-2:]
    ch = _dct_matrix(h, x)
    cw = _dct_matrix(w, x)
    return torch.matmul(torch.matmul(ch.transpose(0, 1), x), cw)


def dct_high_pass(x: torch.Tensor, fraction: float = 0.25) -> torch.Tensor:
    """Remove a square low-frequency region and reconstruct edge cues."""
    if not 0.0 <= fraction < 1.0:
        raise ValueError("fraction must be in [0, 1)")
    coeff = dct_2d(x.float())
    h, w = coeff.shape[-2:]
    side = max(1, min(h, w, int(math.sqrt(h * w * fraction))))
    mask = torch.ones_like(coeff)
    mask[..., :side, :side] = 0
    return idct_2d(coeff * mask).abs().to(dtype=x.dtype)
