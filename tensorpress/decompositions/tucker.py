"""Tucker-2 decomposition for convolutional layers."""

from __future__ import annotations

import logging

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from tensorly.decomposition import partial_tucker

from .base import BaseDecomposition

log = logging.getLogger(__name__)


def _partial_tucker_core_last_first(weight_np: np.ndarray, ranks: list[int]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Call TensorLy ``partial_tucker`` and return ``(core, last_mode0, first_mode1)``.

    TensorLy versions differ: keyword ``rank`` vs ``ranks``, and the return may be
    ``((core, factors), errors)`` or a flatter tuple. This function normalizes.
    """
    modes = [0, 1]
    init = "svd"
    last_err: Exception | None = None
    for kwargs in (
        {"modes": modes, "ranks": ranks, "init": init},
        {"modes": modes, "rank": ranks, "init": init},
    ):
        try:
            out = partial_tucker(weight_np, **kwargs)
            break
        except TypeError as exc:
            last_err = exc
            continue
    else:
        raise last_err if last_err else RuntimeError("partial_tucker failed")

    head = out[0] if isinstance(out, tuple) and len(out) >= 1 else out
    if isinstance(head, tuple) and len(head) == 2:
        core_maybe, factors = head
        if isinstance(core_maybe, np.ndarray) and isinstance(factors, list) and len(factors) >= 2:
            last, first = factors[0], factors[1]
            return core_maybe, last, first
    if isinstance(out, tuple) and len(out) == 2 and isinstance(out[0], np.ndarray):
        core, factors = out[0], out[1]
        if isinstance(factors, (list, tuple)) and len(factors) >= 2:
            last, first = factors[0], factors[1]
            return core, last, first
    raise ValueError(f"Unrecognized partial_tucker return structure: {type(out)}")


class TuckerDecomposition(BaseDecomposition):
    """Tucker-2 decomposition strategy for ``nn.Conv2d`` layers."""

    def decompose(self, layer: nn.Conv2d, ranks: list[int], use_bn: bool = False) -> nn.Sequential:
        """
        Decompose a convolutional layer using Tucker-2 factorization.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.
        ranks : list[int]
            Decomposition ranks ``[R_out, R_in]``.
        use_bn : bool, default=False
            If True, insert ``BatchNorm2d`` after each factorized convolution.

        Returns
        -------
        nn.Sequential
            Factorized replacement module.
        """
        core, last, first = _partial_tucker_core_last_first(layer.weight.data.cpu().numpy(), ranks)

        first_layer = nn.Conv2d(
            in_channels=first.shape[0],
            out_channels=first.shape[1],
            kernel_size=1,
            stride=layer.stride,
            padding=0,
            dilation=layer.dilation,
            bias=False,
        )
        core_layer = nn.Conv2d(
            in_channels=core.shape[1],
            out_channels=core.shape[0],
            kernel_size=layer.kernel_size,
            stride=layer.stride,
            padding=layer.padding,
            dilation=layer.dilation,
            bias=False,
        )
        last_layer = nn.Conv2d(
            in_channels=last.shape[1],
            out_channels=last.shape[0],
            kernel_size=1,
            stride=layer.stride,
            padding=0,
            dilation=layer.dilation,
            bias=(layer.bias is not None),
        )
        if layer.bias is not None:
            last_layer.bias.data = layer.bias.data

        first_t = first.transpose((1, 0))
        first_layer.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(first_t.copy(), axis=-1), axis=-1))
        )
        last_layer.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(last.copy(), axis=-1), axis=-1))
        )
        core_layer.weight.data = torch.from_numpy(np.float32(core.copy()))

        layers: list[nn.Module] = [first_layer, core_layer, last_layer]
        if use_bn:
            layers = [
                first_layer,
                nn.BatchNorm2d(first.shape[1]),
                core_layer,
                nn.BatchNorm2d(core.shape[0]),
                last_layer,
                nn.BatchNorm2d(last.shape[0]),
            ]
        return nn.Sequential(*layers)

    def estimate_ranks(self, layer: nn.Conv2d, compression_factor: float = 0.0) -> list[int]:
        """
        Estimate Tucker ranks with VBMF and optional compression enforcement.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.
        compression_factor : float, default=0.0
            Target compression ratio.

        Returns
        -------
        list[int]
            Tucker ranks ``[R_out, R_in]``.
        """
        from tensorpress._vbmf import EVBMF

        weights = layer.weight.data.cpu().numpy()
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = EVBMF(unfold_0)
        _, diag_1, _, _ = EVBMF(unfold_1)
        # Clamp to >= 1 before any compression step: VBMF can retain zero
        # components for tiny layers, which would make _choose_compression
        # divide by zero.
        ranks = [max(1, int(diag_0.shape[0])), max(1, int(diag_1.shape[1]))]
        log.debug("VBMF estimated Tucker ranks: %s", ranks)

        if compression_factor:
            ranks = self._choose_compression(layer, ranks, compression_factor)
        return [max(1, int(ranks[0])), max(1, int(ranks[1]))]

    @staticmethod
    def _choose_compression(
        layer: nn.Conv2d, ranks: list[int], compression_factor: float = 2.0
    ) -> list[int]:
        """Enforce a minimum compression target for Tucker-2 ranks."""
        weights = layer.weight.data.cpu().numpy()
        t = weights.shape[0]
        s = weights.shape[1]
        d = weights.shape[2]

        compression = (d**2 * s * t) / (s * ranks[0] + ranks[0] * ranks[1] * (d**2) + t * ranks[1])
        ranks[0] = ranks[0] * 3
        if compression <= 2:
            while compression <= compression_factor:
                ranks[0] = ranks[0] // 2
                ranks[1] = ranks[1] // 2
                if ranks[0] < 1 or ranks[1] < 1:
                    ranks[0] = max(1, ranks[0])
                    ranks[1] = max(1, ranks[1])
                    break
                compression = (d**2 * s * t) / (
                    s * ranks[0] + ranks[0] * ranks[1] * (d**2) + t * ranks[1]
                )
        log.debug("Compression factor for layer %s: %s", weights.shape, compression)
        return ranks
