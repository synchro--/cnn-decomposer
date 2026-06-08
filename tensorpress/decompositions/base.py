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

    def compute_ranks(self, layer: Any, compression_ratio: float | None = None) -> list[int]:
        """
        Resolve rank(s) for a layer, dispatching by how the target is specified.

        Single entry point used by the pipeline for non-manual ranks:

        - ``compression_ratio is None`` -> :meth:`estimate_ranks` (statistical,
          data-driven VBMF heuristic that reads the weight values).
        - ``compression_ratio`` given -> :meth:`solve_ranks` (algebraic, exact
          closed-form solve from the layer shape).

        Parameters
        ----------
        layer : Any
            Original convolution layer to compress.
        compression_ratio : float | None, default=None
            Target size reduction as an N-fold factor (``>= 1``). ``None`` means
            "no explicit target, infer the rank from the weights".

        Returns
        -------
        list[int]
            Rank values for the layer.
        """
        if compression_ratio is None:
            return self.estimate_ranks(layer)
        return self.solve_ranks(layer, compression_ratio)

    @abstractmethod
    def estimate_ranks(self, layer: Any) -> list[int]:
        """
        Estimate decomposition rank(s) statistically (VBMF heuristic).

        Reads the weight *values* and infers an intrinsic rank. No size target.

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
    def solve_ranks(self, layer: Any, compression_ratio: float) -> list[int]:
        """
        Solve algebraically for the rank(s) achieving ``compression_ratio``.

        Reads only the layer *shape* and returns the rank that makes the
        factorized layer approximately ``compression_ratio`` times smaller than
        the original (``compression_ratio = original_params / compressed_params``).

        Parameters
        ----------
        layer : Any
            Original convolution layer to compress.
        compression_ratio : float
            Target N-fold size reduction for this layer, ``>= 1``. ``4.0`` means
            "make this layer about 4x smaller".

        Returns
        -------
        list[int]
            Rank values realizing (approximately) the requested ratio.
        """
