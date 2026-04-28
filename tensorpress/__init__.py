"""TensorPress — Tucker / CP decomposition for conv layers."""

from __future__ import annotations

from typing import Any

from tensorpress.config import CompressConfig, FinetuneConfig, LayerSelector, RankEstimator

__all__ = [
    "compress",
    "Compressor",
    "CompressionResult",
    "CompressConfig",
    "FinetuneConfig",
    "LayerSelector",
    "RankEstimator",
]


def __getattr__(name: str) -> Any:
    if name == "Compressor":
        from tensorpress.core.compressor import Compressor

        return Compressor
    if name == "CompressionResult":
        from tensorpress.core.result import CompressionResult

        return CompressionResult
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compress(
    model: Any,
    *,
    method: str = "tucker",
    layers: str | list[str] | None = None,
    ranks: str | float | dict[str, Any] = "auto",
    use_bn: bool = False,
    dataloader: Any | None = None,
    finetune: bool = False,
    finetune_config: FinetuneConfig | None = None,
) -> Any:
    """Convenience wrapper around :class:`Compressor`.

    Uses :class:`CompressConfig` with ``layers=None`` meaning all conv layers when
    ``layers`` is omitted (via ``None`` inside :class:`CompressConfig` defaults).
    """
    from tensorpress.core.compressor import Compressor

    ft = finetune_config if finetune_config is not None else FinetuneConfig()
    cfg = CompressConfig(
        method=method,  # type: ignore[arg-type]
        layers=layers,
        ranks=ranks,  # type: ignore[arg-type]
        use_bn=use_bn,
        finetune=finetune,
        finetune_config=ft,
    )
    return Compressor(cfg).compress(model, dataloader=dataloader)
