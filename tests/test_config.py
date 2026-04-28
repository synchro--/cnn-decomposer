"""Tests for user-facing configuration dataclasses.

Covers defaults (including ``layers=None`` meaning all convs), JSON-style
round-trip for finetune settings, and early validation of invalid compression
method names.
"""

from __future__ import annotations

import pytest

from tensorpress import CompressConfig, FinetuneConfig


def test_compress_config_defaults() -> None:
    """Default :class:`~tensorpress.config.schema.CompressConfig` matches the public contract.

    ``layers is None`` is the canonical \"all supported conv layers\" default;
    method is Tucker and ranks are automatic unless overridden.
    """
    cfg = CompressConfig()
    assert cfg.method == "tucker"
    assert cfg.layers is None
    assert cfg.ranks == "auto"


def test_finetune_config_roundtrip() -> None:
    """``to_dict`` / ``from_dict`` must preserve finetune hyperparameters.

    Ensures configs remain JSON-serializable and reloadable for logging or CLI
    tooling without silent field loss.
    """
    ft = FinetuneConfig(epochs=3, lr=1e-3, use_amp=False)
    back = FinetuneConfig.from_dict(ft.to_dict())
    assert back.epochs == 3
    assert back.lr == pytest.approx(1e-3)


def test_invalid_method() -> None:
    """``validate()`` rejects unknown compression methods before any model work."""
    with pytest.raises(ValueError):
        CompressConfig(method="bad").validate()
