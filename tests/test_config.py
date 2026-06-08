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


def test_compression_ratio_resolves() -> None:
    """A ``compression_ratio`` target resolves to itself; ranks stay 'auto'."""
    cfg = CompressConfig(compression_ratio=4.0)
    cfg.validate()
    assert cfg.effective_compression_ratio() == pytest.approx(4.0)


def test_compression_ratio_out_of_range_rejected() -> None:
    """``compression_ratio`` must be >= 1."""
    with pytest.raises(ValueError):
        CompressConfig(compression_ratio=0.5).validate()
    with pytest.raises(ValueError):
        CompressConfig(compression_ratio=0.0).validate()


def test_compression_ratio_and_blanket_ranks_mutually_exclusive() -> None:
    """A blanket int/list ``ranks`` conflicts with ``compression_ratio``."""
    with pytest.raises(ValueError):
        CompressConfig(compression_ratio=4.0, ranks=8).validate()
    with pytest.raises(ValueError):
        CompressConfig(compression_ratio=4.0, ranks=[8, 8]).validate()


def test_compression_ratio_with_dict_ranks_allowed() -> None:
    """A per-layer ranks dict may coexist with ``compression_ratio`` (override)."""
    cfg = CompressConfig(compression_ratio=4.0, ranks={"layer1": [8, 8]})
    cfg.validate()
    assert cfg.effective_compression_ratio() == pytest.approx(4.0)


def test_explicit_int_ranks_have_no_ratio() -> None:
    """Explicit integer ranks are a distinct (power-user) knob, not a target."""
    cfg = CompressConfig(ranks=8)
    cfg.validate()
    assert cfg.effective_compression_ratio() is None


def test_float_ranks_rejected() -> None:
    """The deprecated float ``ranks`` keep-fraction alias is no longer accepted."""
    with pytest.raises(ValueError):
        CompressConfig(ranks=0.5).validate()
