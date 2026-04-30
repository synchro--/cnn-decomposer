"""High-level :class:`Compressor` API."""

from __future__ import annotations

from dataclasses import replace
from typing import Any

from tensorpress.backends import get_backend
from tensorpress.config import CompressConfig, FinetuneConfig, LayerSelector
from tensorpress.decompositions import CPDecomposition, TuckerDecomposition

from .pipeline import run_pipeline
from .result import CompressedModel

_DECOMP_MAP = {
    "tucker": TuckerDecomposition,
    "cpd": CPDecomposition,
}


class Compressor:
    """Orchestrates layer selection, factorization replacement, and optional finetuning."""

    def __init__(self, config: CompressConfig) -> None:
        config.validate()
        self.config = config
        self._backend = get_backend(config._framework)
        self._decomp = _DECOMP_MAP[config.method]()

    def compress(
        self,
        model: Any,
        dataloader: Any | None = None,
        *,
        finetune: bool | None = None,
        finetune_config: FinetuneConfig | None = None,
    ) -> CompressedModel:
        """Replace selected convolution layers and optionally fine-tune.

        Parameters
        ----------
        finetune, finetune_config
            When provided, override the values stored on :attr:`config` for this run only.
        """
        cfg = self.config
        if finetune is not None or finetune_config is not None:
            cfg = replace(
                self.config,
                finetune=finetune if finetune is not None else self.config.finetune,
                finetune_config=finetune_config
                if finetune_config is not None
                else self.config.finetune_config,
            )
            cfg.validate()

        if cfg.finetune and dataloader is None:
            raise ValueError("dataloader is required when finetune=True")

        selector = LayerSelector(cfg.layers)
        selected = selector.select(model)
        return run_pipeline(model, cfg, self._backend, self._decomp, dataloader, selected)
