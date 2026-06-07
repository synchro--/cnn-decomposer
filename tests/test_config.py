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


def test_compression_keep_fraction_resolves() -> None:
    """A ``compression`` target resolves to a keep fraction; ranks stay 'auto'."""
    cfg = CompressConfig(compression=0.25)
    cfg.validate()
    assert cfg.effective_keep_fraction() == pytest.approx(0.25)


def test_compression_out_of_range_rejected() -> None:
    """``compression`` must lie in (0, 1]."""
    with pytest.raises(ValueError):
        CompressConfig(compression=1.5).validate()
    with pytest.raises(ValueError):
        CompressConfig(compression=0.0).validate()


def test_compression_and_explicit_ranks_mutually_exclusive() -> None:
    """Setting both ``compression`` and explicit ``ranks`` is an error."""
    with pytest.raises(ValueError):
        CompressConfig(compression=0.5, ranks=8).validate()


def test_explicit_int_ranks_have_no_keep_fraction() -> None:
    """Explicit integer ranks are a distinct (power-user) knob, not a target."""
    cfg = CompressConfig(ranks=8)
    cfg.validate()
    assert cfg.effective_keep_fraction() is None


def test_deprecated_float_ranks_is_keep_fraction() -> None:
    """A float ``ranks`` remains accepted as a deprecated keep-fraction alias."""
    cfg = CompressConfig(ranks=0.5)
    cfg.validate()
    assert cfg.effective_keep_fraction() == pytest.approx(0.5)
