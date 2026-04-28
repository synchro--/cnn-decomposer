"""Tests for :class:`~tensorpress.decompositions.cpd.CPDecomposition`.

Validates CP / PARAFAC-style factorization of a conv weight into a depthwise /
pointwise stack: the replacement must match the original conv's output shape for
the same spatial input size.
"""

from __future__ import annotations

import torch

from tensorpress.decompositions import CPDecomposition


def test_cpd_decompose_shape_matches(tiny_conv: torch.nn.Conv2d) -> None:
    """CP replacement must match ``tiny_conv`` output shape for a random input.

    Uses a fixed rank (6) so the decomposition is stable enough for CI without
    depending on VBMF rank estimation.
    """
    decomp = CPDecomposition()
    factorized = decomp.decompose(tiny_conv, 6, use_bn=False)
    x = torch.randn(1, 16, 8, 8)
    assert factorized(x).shape == tiny_conv(x).shape
