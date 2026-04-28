"""Rank-estimation helpers for decomposition configuration."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import tensorly as tl
import torch.nn as nn


class RankEstimator:
    """Resolve user-provided rank specs into concrete rank tuples."""

    def __init__(self, spec: str | float | dict[str, Any]) -> None:
        self.spec = spec

    def estimate(self, name: str, layer: nn.Conv2d) -> tuple[int, ...]:
        """Estimate ranks for a specific layer.

        Parameters
        ----------
        name : str
            Layer name used for per-layer manual rank lookup.
        layer : nn.Conv2d
            Layer to estimate ranks for.

        Returns
        -------
        tuple[int, ...]
            Normalized rank tuple.
        """
        if self.spec == "auto":
            raise ValueError(
                "RankEstimator cannot resolve ranks='auto'; use the active decomposition "
                "strategy via the compression pipeline (decomp.estimate_ranks)."
            )
        if isinstance(self.spec, float):
            return self._from_ratio(layer, self.spec)
        if isinstance(self.spec, dict):
            if name not in self.spec:
                raise ValueError(f"Missing manual rank for layer: {name}")
            return self._normalize_manual(self.spec[name])
        raise ValueError(f"Unsupported rank spec: {self.spec!r}")

    @staticmethod
    def _normalize_manual(value: Any) -> tuple[int, ...]:
        """Normalize manually provided rank values into integer tuple form."""
        if isinstance(value, int):
            if value <= 0:
                raise ValueError("Manual rank int must be > 0")
            return (value,)
        if isinstance(value, tuple):
            if not value:
                raise ValueError("Manual rank tuple must not be empty")
            if not all(isinstance(v, int) and v > 0 for v in value):
                raise ValueError("Manual rank tuple values must be positive ints")
            return value
        if isinstance(value, list):
            if not value:
                raise ValueError("Manual rank list must not be empty")
            if not all(isinstance(v, int) and v > 0 for v in value):
                raise ValueError("Manual rank list values must be positive ints")
            return tuple(value)
        raise ValueError("Manual rank values must be int | list[int] | tuple[int, ...]")

    @staticmethod
    def _from_ratio(layer: nn.Conv2d, ratio: float) -> tuple[int, int]:
        """Estimate Tucker-style ranks from a compression ratio in ``(0, 1]``."""
        if not (0.0 < ratio <= 1.0):
            raise ValueError("Compression ratio must be in range (0, 1]")

        out_channels, in_channels, kernel_h, kernel_w = layer.weight.shape
        kernel_area = int(kernel_h * kernel_w)

        # Approximate Tucker parameters: in*r1 + r1*r0*k^2 + out*r0.
        numerator = ratio * float(out_channels * in_channels * kernel_area)
        denominator = float(in_channels + out_channels + kernel_area)
        base_rank = max(1, int(math.sqrt(max(numerator / max(denominator, 1.0), 1.0))))

        rank_out = max(1, min(base_rank, int(out_channels)))
        rank_in = max(1, min(base_rank, int(in_channels)))
        return (rank_out, rank_in)


class VBMFEstimator:
    """Variational Bayesian Matrix Factorization rank estimator."""

    def estimate(self, layer: nn.Conv2d) -> tuple[int, int]:
        """Estimate Tucker-style ranks using VBMF on unfolded kernels."""
        if not hasattr(layer, "weight"):
            raise ValueError("Layer does not expose a weight tensor")

        weights = layer.weight.data.cpu().numpy()
        if weights.ndim != 4:
            raise ValueError("VBMFEstimator expects a Conv2d-like 4D weight tensor")

        from tensorpress._vbmf import EVBMF

        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = EVBMF(unfold_0)
        _, diag_1, _, _ = EVBMF(unfold_1)

        rank_out = int(diag_0.shape[0])
        rank_in = int(diag_1.shape[0])

        if rank_out <= 0:
            rank_out = max(1, int(np.sqrt(weights.shape[0])))
        if rank_in <= 0:
            rank_in = max(1, int(np.sqrt(weights.shape[1])))

        return (rank_out, rank_in)
