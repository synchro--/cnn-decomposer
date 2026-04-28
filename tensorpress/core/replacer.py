"""Replace convolution layers with factorized modules."""

from __future__ import annotations

import logging
from typing import Any

from tensorpress.backends.base import BaseBackend
from tensorpress.decompositions.base import BaseDecomposition

log = logging.getLogger(__name__)


class LayerReplacer:
    """Swap selected ``Conv2d`` layers with factorized ``nn.Sequential`` replacements."""

    def __init__(self, backend: BaseBackend, decomp: BaseDecomposition, use_bn: bool = False) -> None:
        self.backend = backend
        self.decomp = decomp
        self.use_bn = use_bn

    def replace_layers(
        self,
        model: Any,
        selected_layers: list[tuple[str, Any]],
        rank_map: dict[str, list[int] | int],
    ) -> dict[str, dict[str, Any]]:
        """Decompose each selected layer and install the replacement module.

        Parameters
        ----------
        selected_layers
            ``(dotted_name, conv_layer)`` pairs from :class:`LayerSelector`.
        rank_map
            For each ``dotted_name``, ranks suitable for ``decomp.decompose``:
            ``list[int]`` (Tucker) or ``int`` (CP).
        """
        stats: dict[str, dict[str, Any]] = {}
        for name, layer in selected_layers:
            ranks = rank_map[name]
            log.debug("Decomposing layer '%s' with ranks %s", name, ranks)
            factorized = self.decomp.decompose(layer, ranks, use_bn=self.use_bn)
            self.backend.replace_module(model, name, factorized)
            orig = sum(p.numel() for p in layer.parameters())
            compr = sum(p.numel() for p in factorized.parameters())
            ratio = orig / max(compr, 1)
            stats[name] = {
                "original_params": orig,
                "compressed_params": compr,
                "ranks": ranks,
                "ratio": ratio,
            }
            log.debug("Compression ratio for '%s': %.2fx", name, ratio)
        return stats
