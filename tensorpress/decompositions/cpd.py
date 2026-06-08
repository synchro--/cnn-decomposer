"""CP decomposition for convolutional layers."""

from __future__ import annotations

import logging

import numpy as np
import tensorly as tl
import torch
import torch.nn as nn
from tensorly.decomposition import parafac

from .base import BaseDecomposition

log = logging.getLogger(__name__)


class CPDecomposition(BaseDecomposition):
    """CP decomposition strategy for ``nn.Conv2d`` layers."""

    def decompose(self, layer: nn.Conv2d, ranks: int | list[int], use_bn: bool = False) -> nn.Sequential:
        """
        Decompose a convolutional layer with CANDECOMP/PARAFAC factors.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.
        ranks : int | list[int]
            Decomposition rank or rank wrapped in a list.
        use_bn : bool, default=False
            If True, insert ``BatchNorm2d`` after each factorized convolution.

        Returns
        -------
        nn.Sequential
            Factorized replacement module.
        """
        x = layer.weight.data.cpu().numpy()
        rank = ranks if isinstance(ranks, int) else ranks[0]

        init = "random" if max(x.shape) >= 256 else "svd"
        _, factors = parafac(x, rank=rank, init=init)
        last, first, vertical, horizontal = factors

        first_pointwise = nn.Conv2d(
            in_channels=first.shape[0],
            out_channels=first.shape[1],
            kernel_size=1,
            stride=layer.stride,
            padding=0,
            dilation=layer.dilation,
            bias=False,
        )
        depthwise_vertical = nn.Conv2d(
            in_channels=vertical.shape[1],
            out_channels=vertical.shape[1],
            kernel_size=(vertical.shape[0], 1),
            stride=layer.stride,
            padding=(layer.padding[0], 0),
            dilation=layer.dilation,
            groups=vertical.shape[1],
            bias=False,
        )
        depthwise_horizontal = nn.Conv2d(
            in_channels=horizontal.shape[1],
            out_channels=horizontal.shape[1],
            kernel_size=(1, horizontal.shape[0]),
            stride=layer.stride,
            padding=(0, layer.padding[0]),
            dilation=layer.dilation,
            groups=horizontal.shape[1],
            bias=False,
        )
        last_pointwise = nn.Conv2d(
            in_channels=last.shape[1],
            out_channels=last.shape[0],
            kernel_size=1,
            stride=layer.stride,
            padding=0,
            dilation=layer.dilation,
            bias=(layer.bias is not None),
        )
        if layer.bias is not None:
            last_pointwise.bias.data = layer.bias.data

        depthwise_vertical.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(vertical.transpose(1, 0), axis=1), axis=-1))
        )
        depthwise_horizontal.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(horizontal.transpose(1, 0), axis=1), axis=1))
        )
        first_pointwise.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(first.transpose(1, 0), axis=-1), axis=-1))
        )
        last_pointwise.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(last, axis=-1), axis=-1))
        )

        layers: list[nn.Module] = [
            first_pointwise,
            depthwise_vertical,
            depthwise_horizontal,
            last_pointwise,
        ]
        if use_bn:
            layers = [
                first_pointwise,
                nn.BatchNorm2d(first_pointwise.out_channels),
                depthwise_vertical,
                nn.BatchNorm2d(depthwise_vertical.out_channels),
                depthwise_horizontal,
                nn.BatchNorm2d(depthwise_horizontal.out_channels),
                last_pointwise,
                nn.BatchNorm2d(last_pointwise.out_channels),
            ]
        return nn.Sequential(*layers)

    def estimate_ranks(self, layer: nn.Conv2d) -> list[int]:
        """
        Estimate CP rank automatically with VBMF on the kernel unfoldings.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.

        Returns
        -------
        list[int]
            Estimated rank in ``[R]`` form.
        """
        from tensorpress._vbmf import EVBMF

        weights = layer.weight.data.cpu().numpy()
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = EVBMF(unfold_0)
        _, diag_1, _, _ = EVBMF(unfold_1)
        rank = max(diag_0.shape[0], diag_1.shape[0])
        if rank == 0:
            rank = 10

        log.debug("VBMF estimated CP rank: %d", rank)
        return [max(int(rank), 1)]

    def solve_ranks(self, layer: nn.Conv2d, compression_ratio: float) -> list[int]:
        """
        Solve for the CP rank achieving an N-fold ``compression_ratio``.

        For a CP-factorized conv the parameter count is approximately
        ``R * (S + Kh + Kw + T)`` (the two pointwise plus two depthwise convs),
        while the original conv has ``T * S * Kh * Kw`` parameters. Setting
        ``original / compressed = compression_ratio`` and solving for ``R`` gives
        the returned rank.

        Notes
        -----
        Because CP's per-rank cost ``(S + Kh + Kw + T)`` is small relative to the
        dense kernel, a *low* ratio (close to 1x) requires a large rank (and a
        correspondingly expensive PARAFAC fit). Larger ratios yield smaller ranks
        and far lower decomposition memory/time.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.
        compression_ratio : float
            Target N-fold size reduction, ``>= 1``.

        Returns
        -------
        list[int]
            CP rank in ``[R]`` form.
        """
        if compression_ratio < 1.0:
            raise ValueError("compression_ratio must be >= 1")

        out_ch, in_ch, kh, kw = (int(v) for v in layer.weight.shape)
        original = out_ch * in_ch * kh * kw
        per_rank = in_ch + kh + kw + out_ch
        rank = int(round(original / (compression_ratio * max(per_rank, 1))))
        rank = max(rank, 1)
        log.debug(
            "CP compression_ratio=%.2fx -> rank=%d (shape=%s)",
            compression_ratio,
            rank,
            layer.weight.shape,
        )
        return [rank]
