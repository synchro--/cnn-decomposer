"""Semantic tests for the N-fold ``compression_ratio`` target.

These cover the bug that previously went undetected: a requested compression
target that was silently ignored (the rank was never actually capped). The old
suite only asserted ``len(ranks) == 2`` and ``rank >= 1``, which a no-op capping
function passes trivially. Here we assert that the *requested* N-fold ratio is
actually *realized* on the compressed layer — for both Tucker and CP — and that
the relationship is monotonic (larger ratio -> fewer params), plus the per-layer
resolution precedence between a manual ``ranks`` dict and a blanket ratio.
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


@pytest.mark.parametrize("ratio", [2.0, 4.0, 10.0])
@pytest.mark.parametrize(
    ("decomp_cls", "params_fn"),
    [(TuckerDecomposition, _tucker_factorized_params), (CPDecomposition, _cp_factorized_params)],
)
def test_compression_ratio_is_realized(decomp_cls, params_fn, ratio: float) -> None:
    """Requested N-fold ratio must be approximately realized on the layer.

    This is the regression guard: if rank capping is a no-op (the original CPD
    bug), the realized ratio stays near 1.0 and this assertion fails.
    """
    conv = nn.Conv2d(64, 96, kernel_size=3, padding=1, bias=False)
    original = sum(p.numel() for p in conv.parameters())

    ranks = decomp_cls().solve_ranks(conv, ratio)
    realized = original / params_fn(conv, ranks)

    # Tolerance scales with the target (rounding to integer ranks).
    assert realized == pytest.approx(ratio, rel=0.2)


@pytest.mark.parametrize(
    ("decomp_cls", "params_fn"),
    [(TuckerDecomposition, _tucker_factorized_params), (CPDecomposition, _cp_factorized_params)],
)
def test_compression_ratio_is_monotonic(decomp_cls, params_fn) -> None:
    """A larger ratio target must yield strictly fewer factorized parameters."""
    conv = nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False)
    decomp = decomp_cls()

    aggressive = params_fn(conv, decomp.solve_ranks(conv, 10.0))
    gentle = params_fn(conv, decomp.solve_ranks(conv, 2.0))

    assert aggressive < gentle


def test_compute_ranks_dispatches() -> None:
    """compute_ranks routes to solve_ranks with a ratio and estimate_ranks without."""
    conv = nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False)
    decomp = TuckerDecomposition()

    assert decomp.compute_ranks(conv, compression_ratio=4.0) == decomp.solve_ranks(conv, 4.0)
    assert decomp.compute_ranks(conv) == decomp.estimate_ranks(conv)


@pytest.mark.parametrize("method", ["tucker", "cpd"])
def test_compressor_reports_realized_ratio(method: str, tiny_model: nn.Module) -> None:
    """End-to-end: ``compression_ratio`` shrinks the subset and reports req-vs-realized."""
    cfg = CompressConfig(method=method, layers="all", compression_ratio=2.0)
    result = Compressor(cfg).compress(tiny_model)

    assert result.requested_compression_ratio == pytest.approx(2.0)
    assert result.compressed_params_after < result.compressed_params_before
    assert result.subset_compression_ratio > 1.0
    # Reporting must run without error and not blend scopes silently.
    result.report()
    summary = result.compare()
    assert summary["compressed_subset"]["requested_compression_ratio"] == pytest.approx(2.0)


def test_tiny_model_forward_after_compression(tiny_model: nn.Module) -> None:
    """The compressed model must remain a drop-in for the original I/O shape."""
    x = torch.randn(1, 3, 8, 8)
    expected = tiny_model(x).shape
    result = Compressor(
        CompressConfig(method="tucker", layers="all", compression_ratio=2.0)
    ).compress(tiny_model)
    assert result(x).shape == expected


def test_dict_override_takes_precedence_over_ratio() -> None:
    """A per-layer ranks dict entry overrides the blanket compression_ratio."""
    model = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
    )
    cfg = CompressConfig(
        method="tucker",
        layers="all",
        ranks={"0": [4, 4]},
        compression_ratio=2.0,
    )
    result = Compressor(cfg).compress(model)

    # The overridden layer "0" uses the manual rank [4, 4]; layer "2" uses ratio.
    assert result.layer_stats["0"]["ranks"] == [4, 4]
    assert "2" in result.layer_stats


def test_dict_alone_restricts_selection() -> None:
    """A ranks dict with no explicit layers/ratio only touches the dict keys."""
    model = nn.Sequential(
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
        nn.ReLU(),
        nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
    )
    cfg = CompressConfig(method="tucker", ranks={"0": [4, 4]})
    result = Compressor(cfg).compress(model)

    assert set(result.layer_stats) == {"0"}
