"""Integration tests for the top-level :func:`~tensorpress.compress` API.

Ensures the convenience entrypoint builds a :class:`~tensorpress.core.compressor.Compressor`,
runs compression on a trivial ``Sequential`` of convs, and that
:class:`~tensorpress.core.result.CompressedModel` reporting helpers run without
raising (plain text if Rich is absent).
"""

from __future__ import annotations

import torch
import torch.nn as nn

from tensorpress import compress


def test_compress_function_smoke() -> None:
    """``compress(...)`` should modify the model in place and preserve I/O shapes.

    Uses ``Sequential`` indices ``\"0\"`` and ``\"1\"`` as layer keys in the
    rank dict. Calls ``report()`` and ``compare()`` to smoke-test the result
    object’s user-facing output paths.
    """
    model = nn.Sequential(
        nn.Conv2d(3, 4, 3, padding=1),
        nn.Conv2d(4, 4, 3, padding=1),
    )
    ranks = {"0": [2, 2], "1": [2, 2]}
    res = compress(model, method="tucker", ranks=ranks, use_bn=False)
    x = torch.randn(1, 3, 8, 8)
    y = model(x)
    assert y.shape == (1, 4, 8, 8)
    res.report()
    summary = res.compare()
    assert isinstance(summary, dict)
    assert "compression_ratio" in summary["whole_model"]
    assert "compression_ratio" in summary["compressed_subset"]
    out = res(torch.randn(1, 3, 8, 8))
    assert out.shape == (1, 4, 8, 8)
