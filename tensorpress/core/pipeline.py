"""End-to-end compression orchestration (no direct ``torch`` imports here)."""

from __future__ import annotations

import logging
from typing import Any

from tensorpress.backends.base import BaseBackend
from tensorpress.config import CompressConfig, RankEstimator
from tensorpress.decompositions.base import BaseDecomposition

from .finetuner import FineTuner
from .replacer import LayerReplacer
from .result import CompressedModel

log = logging.getLogger(__name__)


def _resolve_ranks_list(
    config: CompressConfig,
    layer_name: str,
    layer: Any,
    decomp: BaseDecomposition,
) -> list[int]:
    """Return ranks as a flat list of positive integers for normalization."""
    spec = config.ranks
    if isinstance(spec, dict):
        est = RankEstimator(spec).estimate(layer_name, layer)
        return list(est)
    if spec == "auto":
        return decomp.estimate_ranks(layer, compression_factor=0.0)
    if isinstance(spec, float):
        return decomp.estimate_ranks(layer, compression_factor=spec)
    raise ValueError(f"unsupported ranks configuration: {spec!r}")


def _ranks_for_decompose(method: str, ranks: list[int]) -> list[int] | int:
    """Adapt ``list`` form to what each decomposition expects."""
    if method == "tucker":
        if len(ranks) == 1:
            return [ranks[0], ranks[0]]
        if len(ranks) == 2:
            return ranks
        raise ValueError("Tucker ranks must provide one or two positive integers per layer")
    if method == "cpd":
        if len(ranks) < 1:
            raise ValueError("CP ranks must provide at least one integer per layer")
        return ranks[0]
    raise ValueError(f"unknown method: {method}")


def run_pipeline(
    model: Any,
    config: CompressConfig,
    backend: BaseBackend,
    decomp: BaseDecomposition,
    dataloader: Any | None,
    selected_layers: list[tuple[str, Any]],
) -> CompressedModel:
    """Run replacement and optional finetuning; return a :class:`CompressedModel`."""
    before = backend.count_parameters(model)

    rank_map: dict[str, list[int] | int] = {}
    for name, layer in selected_layers:
        raw = _resolve_ranks_list(config, name, layer, decomp)
        rank_map[name] = _ranks_for_decompose(config.method, raw)

    replacer = LayerReplacer(backend, decomp, use_bn=config.use_bn)
    layer_stats = replacer.replace_layers(model, selected_layers, rank_map)
    after = backend.count_parameters(model)

    history: list[float] | None = None
    if config.finetune:
        if dataloader is None:
            raise ValueError("dataloader is required when finetune=True")
        tuner = FineTuner(config.finetune_config, backend)
        history = tuner.finetune(model, dataloader)
        log.debug("Finetune complete; %d epochs", len(history))

    return CompressedModel(
        model=model,
        layer_stats=layer_stats,
        trainable_params_before=before,
        trainable_params_after=after,
        finetune_history=history,
        backend=backend,
    )
