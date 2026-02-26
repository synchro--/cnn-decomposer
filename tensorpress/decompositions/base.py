"""Abstract interfaces for decomposition backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseDecomposition(ABC):
    """Abstract base class for tensor decomposition strategies."""

    @abstractmethod
    def decompose(self, layer: Any, ranks: Any, use_bn: bool = False) -> Any:
        """
        Decompose a convolutional layer into a factorized replacement.

        Parameters
        ----------
        layer : Any
            Original convolution layer to compress.
        ranks : Any
            Rank(s) for the decomposition.
        use_bn : bool, default=False
            If True, insert batch normalization layers between factors.

        Returns
        -------
        Any
            Drop-in replacement module, typically an ``nn.Sequential``.
        """

    @abstractmethod
    def estimate_ranks(self, layer: Any, compression_factor: float = 0.0) -> list[int]:
        """
        Estimate decomposition rank(s), optionally enforcing compression.

        Parameters
        ----------
        layer : Any
            Original convolution layer to compress.
        compression_factor : float, default=0.0
            Desired compression ratio. If 0, keep raw VBMF estimates.

        Returns
        -------
        list[int]
            Estimated rank values.
        """
