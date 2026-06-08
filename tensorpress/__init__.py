"""TensorPress — Tucker / CP decomposition for conv layers."""

from __future__ import annotations

from typing import Any

from tensorpress.config import CompressConfig, FinetuneConfig, LayerSelector, RankEstimator
from tensorpress.core.result import CompressedModel

__all__ = [
    "compress",
    "Compressor",
    "CompressedModel",
    "CompressConfig",
    "FinetuneConfig",
    "LayerSelector",
    "RankEstimator",
]


def __getattr__(name: str) -> Any:
    if name == "Compressor":
        from tensorpress.core.compressor import Compressor

        return Compressor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def compress(
    model: Any,
    *,
    method: str = "tucker",
    layers: str | list[str] | None = None,
    ranks: str | int | list[int] | dict[str, Any] = "auto",
    compression_ratio: float | None = None,
    use_bn: bool = False,
    dataloader: Any | None = None,
    finetune: bool = False,
    finetune_config: FinetuneConfig | None = None,
) -> CompressedModel:
    """Convenience wrapper around :class:`Compressor`.

    Uses :class:`CompressConfig` with ``layers=None`` meaning all conv layers when
    ``layers`` is omitted (via ``None`` inside :class:`CompressConfig` defaults).

    Returns
    -------
    CompressedModel
        Wrapper around the compressed model with reporting and PyTorch proxies.
    """
    from tensorpress.core.compressor import Compressor

    ft = finetune_config if finetune_config is not None else FinetuneConfig()
    cfg = CompressConfig(
        method=method,  # type: ignore[arg-type]
        layers=layers,
        ranks=ranks,  # type: ignore[arg-type]
        compression_ratio=compression_ratio,
        use_bn=use_bn,
        finetune=finetune,
        finetune_config=ft,
    )
    return Compressor(cfg).compress(model, dataloader=dataloader)
