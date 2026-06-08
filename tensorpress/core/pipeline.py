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
    """Return ranks as a flat list of positive integers for normalization.

    Per-layer precedence:

    1. a manual ``ranks`` dict entry for this layer (highest priority);
    2. a blanket ``compression_ratio`` target (algebraic ``solve_ranks``);
    3. a blanket ``int``/``list`` manual rank;
    4. ``"auto"`` (or a dict that does not cover this layer) -> ``estimate_ranks``.
    """
    spec = config.ranks
    ratio = config.effective_compression_ratio()

    # 1. per-layer manual override.
    if isinstance(spec, dict) and layer_name in spec:
        return list(RankEstimator(spec).estimate(layer_name, layer))

    # 2. blanket compression-ratio target.
    if ratio is not None:
        return decomp.compute_ranks(layer, compression_ratio=ratio)

    # 3. blanket manual int/list.
    if isinstance(spec, bool):
        raise ValueError(f"unsupported ranks configuration: {spec!r}")
    if isinstance(spec, int):
        return [spec]
    if isinstance(spec, (list, tuple)):
        return list(spec)

    # 4. auto estimate (also covers a dict with no entry for this layer).
    if spec == "auto" or isinstance(spec, dict):
        return decomp.compute_ranks(layer)
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
        requested_compression_ratio=config.effective_compression_ratio(),
    )
