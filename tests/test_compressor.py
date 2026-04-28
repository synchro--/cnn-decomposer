"""Tests for :class:`~tensorpress.core.compressor.Compressor` orchestration.

Uses explicit per-layer Tucker ranks (dict) so tests do not depend on VBMF or
``ranks=\"auto\"``. Verifies in-place replacement, non-empty layer stats, and
that the model still runs after compression.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tensorpress import CompressConfig, Compressor


def test_compressor_tucker_manual_ranks(tiny_model: nn.Module) -> None:
    """End-to-end compress: select convs by name, replace with Tucker factors, run forward.

    - ``ranks`` maps ``blocks.0`` / ``blocks.2`` to small Tucker ranks for speed.
    - ``result.layer_stats`` must be populated (one entry per replaced layer).
    - Parameter count may change slightly with tiny ranks; shape ``(1,16,16,16)``
      checks the nominal data layout is preserved (channels × height × width).
    """
    cfg = CompressConfig(
        method="tucker",
        layers=None,
        ranks={
            "blocks.0": [2, 2],
            "blocks.2": [2, 2],
        },
        use_bn=False,
        finetune=False,
    )
    comp = Compressor(cfg)
    x = torch.randn(1, 3, 16, 16)
    before = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
    result = comp.compress(tiny_model)
    after = sum(p.numel() for p in tiny_model.parameters() if p.requires_grad)
    assert result.layer_stats
    assert after <= before + 1
    assert tiny_model(x).shape == (1, 16, 16, 16)
