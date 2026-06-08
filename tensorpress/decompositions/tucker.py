"""Tucker-2 decomposition for convolutional layers."""

from __future__ import annotations

import logging
import math

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

    def estimate_ranks(self, layer: nn.Conv2d) -> list[int]:
        """
        Estimate Tucker ranks automatically with VBMF on the kernel unfoldings.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.

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
        # Clamp to >= 1: VBMF can retain zero components for tiny layers.
        ranks = [max(1, int(diag_0.shape[0])), max(1, int(diag_1.shape[1]))]
        log.debug("VBMF estimated Tucker ranks: %s", ranks)
        return ranks

    def solve_ranks(self, layer: nn.Conv2d, compression_ratio: float) -> list[int]:
        """
        Solve for Tucker ranks ``[R_out, R_in]`` achieving ``compression_ratio``.

        A Tucker-2 conv has approximately ``S*R_in + R_in*R_out*Kh*Kw + R_out*T``
        parameters versus the original ``T*S*Kh*Kw``. We scale both ranks by a
        single factor ``alpha`` proportional to their mode sizes
        (``R_out = alpha*T``, ``R_in = alpha*S``), which reduces the budget
        equation ``original / compressed = compression_ratio`` to a quadratic in
        ``alpha`` that we solve in closed form.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.
        compression_ratio : float
            Target N-fold size reduction, ``>= 1``.

        Returns
        -------
        list[int]
            Tucker ranks ``[R_out, R_in]``.
        """
        if compression_ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1")

        out_ch, in_ch, kh, kw = (int(v) for v in layer.weight.shape)
        spatial = kh * kw

        # a*alpha^2 + b*alpha - c = 0, with R_out=alpha*out, R_in=alpha*in,
        # and compressed budget c = original / compression_ratio.
        a = float(in_ch * out_ch * spatial)
        b = float(in_ch * in_ch + out_ch * out_ch)
        c = float(out_ch * in_ch * spatial) / compression_ratio
        alpha = (-b + math.sqrt(b * b + 4.0 * a * c)) / (2.0 * a)

        rank_out = max(1, min(out_ch, int(round(alpha * out_ch))))
        rank_in = max(1, min(in_ch, int(round(alpha * in_ch))))
        log.debug(
            "Tucker compression_ratio=%.2fx -> ranks=[%d, %d] (shape=%s)",
            compression_ratio,
            rank_out,
            rank_in,
            layer.weight.shape,
        )
        return [rank_out, rank_in]
