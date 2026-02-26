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

    def estimate_ranks(self, layer: nn.Conv2d, compression_factor: float = 0.0) -> list[int]:
        """
        Estimate CP rank with VBMF and optional compression enforcement.

        Parameters
        ----------
        layer : nn.Conv2d
            Input convolution layer.
        compression_factor : float, default=0.0
            Target compression ratio.

        Returns
        -------
        list[int]
            Estimated rank in ``[R]`` form.
        """
        from VBMF import VBMF

        weights = layer.weight.data.cpu().numpy()
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        rank = max(diag_0.shape[0], diag_1.shape[1])
        if rank == 0:
            rank = 10
        ranks = [rank, rank]

        if compression_factor:
            ranks = self._choose_compression(layer, ranks, compression_factor)
            rank = int(ranks[0])

        log.debug("VBMF estimated CP rank: %d", rank)
        return [max(int(rank), 1)]

    @staticmethod
    def _choose_compression(
        layer: nn.Conv2d, ranks: list[int], compression_factor: float = 2.0
    ) -> list[int]:
        """Enforce a minimum compression target for CP rank."""
        weights = layer.weight.data.cpu().numpy()
        t = weights.shape[0]
        s = weights.shape[1]
        d = weights.shape[2]

        rank = ranks[0]
        compression = (d**2 * t * s) / (rank * (s + 2 * d + t))
        if compression <= compression_factor:
            rank = (d**2 * s * t) / (compression_factor * (s + 2 * d + t))
            ranks[0] = max(int(np.floor(rank)), 1)
            ranks[1] = ranks[0]
            compression = (d**2 * s * t) / (max(rank, 1.0) * (s + 2 * d + t))
        log.debug("Compression factor for layer %s: %s", weights.shape, compression)
        return ranks
