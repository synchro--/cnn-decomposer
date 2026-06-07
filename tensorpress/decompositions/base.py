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
    def estimate_ranks(self, layer: Any) -> list[int]:
        """
        Estimate decomposition rank(s) automatically (VBMF heuristic).

        Parameters
        ----------
        layer : Any
            Original convolution layer to compress.

        Returns
        -------
        list[int]
            Estimated rank values.
        """

    @abstractmethod
    def ranks_for_keep_fraction(self, layer: Any, keep: float) -> list[int]:
        """
        Solve for the rank(s) that retain ``keep`` fraction of the layer params.

        Parameters
        ----------
        layer : Any
            Original convolution layer to compress.
        keep : float
            Target fraction of this layer's parameters to keep, in (0, 1].
            ``0.25`` keeps roughly a quarter of the parameters.

        Returns
        -------
        list[int]
            Rank values realizing (approximately) the requested keep fraction.
        """
