"""Layer selection utilities for compression configuration."""

from __future__ import annotations

import re
from typing import Any

import torch.nn as nn

from .schema import LayerSpec


class LayerSelector:
    """Resolve a layer-selection spec into concrete ``(name, module)`` pairs."""

    def __init__(self, spec: LayerSpec) -> None:
        self.spec = spec

    def select(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Return convolution layers matching this selector.

        Parameters
        ----------
        model : nn.Module
            Model to inspect.

        Returns
        -------
        list[tuple[str, nn.Module]]
            Matching ``(name, module)`` pairs for ``nn.Conv2d`` layers.
        """
        if self.spec is None or self.spec == "all":
            return self._all_conv(model)
        if isinstance(self.spec, list):
            return self._by_names(model, self.spec)
        if isinstance(self.spec, str):
            return self._by_regex(model, self.spec)
        if callable(self.spec):
            return self._by_predicate(model, self.spec)
        raise ValueError(f"Unknown layer spec: {self.spec!r}")

    @staticmethod
    def _iter_conv(model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Collect all named convolution layers in declaration order."""
        return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Conv2d)]

    def _all_conv(self, model: nn.Module) -> list[tuple[str, nn.Module]]:
        """Return all convolution layers from the model."""
        return self._iter_conv(model)

    def _by_names(self, model: nn.Module, names: list[str]) -> list[tuple[str, nn.Module]]:
        """Return convolution layers that exactly match the provided names."""
        conv_layers = dict(self._iter_conv(model))
        selected: list[tuple[str, nn.Module]] = []
        missing: list[str] = []
        for name in names:
            layer = conv_layers.get(name)
            if layer is None:
                missing.append(name)
                continue
            selected.append((name, layer))

        if missing:
            available = ", ".join(conv_layers.keys()) or "<none>"
            missing_fmt = ", ".join(missing)
            raise ValueError(
                f"Unknown convolution layer name(s): {missing_fmt}. "
                f"Available Conv2d layers: {available}"
            )
        return selected

    def _by_regex(self, model: nn.Module, pattern: str) -> list[tuple[str, nn.Module]]:
        """Return convolution layers where the name matches a regex."""
        try:
            compiled = re.compile(pattern)
        except re.error as exc:
            raise ValueError(f"Invalid regex layer selector: {pattern!r}") from exc

        return [(name, module) for name, module in self._iter_conv(model) if compiled.search(name)]

    def _by_predicate(
        self,
        model: nn.Module,
        predicate: Any,
    ) -> list[tuple[str, nn.Module]]:
        """Return convolution layers accepted by a user predicate."""
        selected: list[tuple[str, nn.Module]] = []
        for name, module in self._iter_conv(model):
            if predicate(name, module):
                selected.append((name, module))
        return selected
