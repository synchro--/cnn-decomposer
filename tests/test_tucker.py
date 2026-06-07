"""Tests for :class:`~tensorpress.decompositions.tucker.TuckerDecomposition`.

These tests focus on mathematical/structural correctness of the Tucker-2
replacement: output shapes must match the original layer, and the factorized
stack should not explode in parameter count for moderate ranks.
"""

from __future__ import annotations

import torch

from tensorpress.decompositions import TuckerDecomposition


def test_tucker_decompose_shape_matches(tiny_conv: torch.nn.Conv2d) -> None:
    """Factorized ``Sequential`` must produce the same output shape as ``tiny_conv``.

    If shapes diverge, the replacement is not a drop-in for that conv (wrong
    channel geometry or stride/padding handling).
    """
    decomp = TuckerDecomposition()
    ranks = [4, 4]
    factorized = decomp.decompose(tiny_conv, ranks, use_bn=False)
    x = torch.randn(1, 16, 8, 8)
    assert factorized(x).shape == tiny_conv(x).shape


def test_tucker_fewer_or_equal_params(tiny_conv: torch.nn.Conv2d) -> None:
    """With ranks ``[8, 8]``, total parameters should not grow much vs the original.

    Allows a small slack (5%) for rounding/layout overhead; catches accidental
    duplication of large tensors in the factorized modules.
    """
    decomp = TuckerDecomposition()
    factorized = decomp.decompose(tiny_conv, [8, 8], use_bn=False)
    orig = sum(p.numel() for p in tiny_conv.parameters())
    compr = sum(p.numel() for p in factorized.parameters())
    assert compr <= orig * 1.05


def test_estimate_ranks_single_input_channel() -> None:
    """VBMF auto-rank estimation must work for a 1-input-channel conv.

    The mode-0 unfolding of a ``Conv2d(1, 32, 3)`` weight is tall (``32 x 9``),
    violating EVBMF's documented ``L <= M`` precondition. Without the transpose
    guard this raised ``ValueError: The lower bound exceeds the upper bound``
    (the fashion-mnist/mnist quickstart crash).
    """
    conv = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
    torch.nn.init.normal_(conv.weight)
    ranks = TuckerDecomposition().estimate_ranks(conv)
    assert len(ranks) == 2
    assert all(r >= 1 for r in ranks)


def test_keep_fraction_single_input_channel() -> None:
    """The keep-fraction path must also work for a 1-channel conv."""
    conv = torch.nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False)
    torch.nn.init.normal_(conv.weight)
    ranks = TuckerDecomposition().ranks_for_keep_fraction(conv, 0.5)
    assert len(ranks) == 2
    assert all(r >= 1 for r in ranks)
