"""Semantic tests for the ``compression`` keep-fraction target.

These cover the bug that previously went undetected: a requested compression
target that was silently ignored (the rank was never actually capped). The old
suite only asserted ``len(ranks) == 2`` and ``rank >= 1``, which a no-op capping
function passes trivially. Here we assert that the *requested* keep fraction is
actually *realized* on the compressed layer — for both Tucker and CP — and that
the relationship is monotonic (smaller target -> smaller rank/params).
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from tensorpress import CompressConfig, Compressor
from tensorpress.decompositions import CPDecomposition, TuckerDecomposition


def _cp_factorized_params(layer: nn.Conv2d, ranks: list[int]) -> int:
    """Parameter count of the CP replacement implied by ``ranks`` (no bias)."""
    out_ch, in_ch, kh, kw = (int(v) for v in layer.weight.shape)
    rank = ranks[0]
    return rank * (in_ch + kh + kw + out_ch)


def _tucker_factorized_params(layer: nn.Conv2d, ranks: list[int]) -> int:
    """Parameter count of the Tucker replacement implied by ``ranks`` (no bias)."""
    out_ch, in_ch, kh, kw = (int(v) for v in layer.weight.shape)
    rank_out, rank_in = ranks
    return in_ch * rank_in + rank_in * rank_out * kh * kw + rank_out * out_ch


@pytest.mark.parametrize("keep", [0.1, 0.25, 0.5])
@pytest.mark.parametrize(
    ("decomp_cls", "params_fn"),
    [(TuckerDecomposition, _tucker_factorized_params), (CPDecomposition, _cp_factorized_params)],
)
def test_keep_fraction_is_realized(decomp_cls, params_fn, keep: float) -> None:
    """Requested keep fraction must be approximately realized on the layer.

    This is the regression guard: if rank capping is a no-op (the original CPD
    bug), the realized fraction stays near 1.0 and this assertion fails.
    """
    conv = nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False)
    original = sum(p.numel() for p in conv.parameters())

    ranks = decomp_cls().ranks_for_keep_fraction(conv, keep)
    realized = params_fn(conv, ranks) / original

    assert realized == pytest.approx(keep, abs=0.05)


@pytest.mark.parametrize(
    ("decomp_cls", "params_fn"),
    [(TuckerDecomposition, _tucker_factorized_params), (CPDecomposition, _cp_factorized_params)],
)
def test_keep_fraction_is_monotonic(decomp_cls, params_fn) -> None:
    """A smaller keep target must yield strictly fewer factorized parameters."""
    conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
    decomp = decomp_cls()

    small = params_fn(conv, decomp.ranks_for_keep_fraction(conv, 0.1))
    large = params_fn(conv, decomp.ranks_for_keep_fraction(conv, 0.5))

    assert small < large


@pytest.mark.parametrize("method", ["tucker", "cpd"])
def test_compressor_reports_realized_keep_fraction(method: str, tiny_model: nn.Module) -> None:
    """End-to-end: ``compression`` shrinks the subset and reports requested-vs-realized."""
    cfg = CompressConfig(method=method, layers="all", compression=0.5)
    result = Compressor(cfg).compress(tiny_model)

    assert result.requested_keep_fraction == pytest.approx(0.5)
    assert result.compressed_params_after < result.compressed_params_before
    assert result.subset_compression_ratio > 1.0
    # Reporting must run without error and not blend scopes silently.
    result.report()
    summary = result.compare()
    assert summary["compressed_subset"]["requested_keep_fraction"] == pytest.approx(0.5)


def test_tiny_model_forward_after_keep_fraction(tiny_model: nn.Module) -> None:
    """The compressed model must remain a drop-in for the original I/O shape."""
    x = torch.randn(1, 3, 8, 8)
    expected = tiny_model(x).shape
    result = Compressor(
        CompressConfig(method="tucker", layers="all", compression=0.5)
    ).compress(tiny_model)
    assert result(x).shape == expected
