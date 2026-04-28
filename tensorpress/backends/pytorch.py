"""PyTorch backend implementation."""

from __future__ import annotations

import logging
from typing import Any, Iterator

import torch
import torch.nn as nn

from .base import BaseBackend

log = logging.getLogger(__name__)


class PyTorchBackend(BaseBackend):
    """PyTorch-specific helpers for layer discovery, replacement, and I/O."""

    def get_conv_layers(self, model: nn.Module) -> Iterator[tuple[str, nn.Module]]:
        """Yield ``(name, Conv2d)`` for every ``nn.Conv2d`` in declaration order."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                yield name, module

    def replace_module(self, model: nn.Module, name: str, new_module: nn.Module) -> None:
        """Replace ``name`` (dotted path, including numeric ``Sequential`` indices)."""
        parts = name.split(".")
        parent: Any = model
        for part in parts[:-1]:
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        last = parts[-1]
        if last.isdigit():
            parent[int(last)] = new_module
        else:
            setattr(parent, last, new_module)
        log.debug("Replaced module %s", name)

    def count_parameters(self, model: nn.Module) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_model(self, model: nn.Module, path: str) -> None:
        """Save ``state_dict`` to ``path``."""
        torch.save(model.state_dict(), path)
        log.debug("Saved state dict to %s", path)

    def load_model(self, model: nn.Module, path: str) -> nn.Module:
        """Load ``state_dict`` from ``path`` into ``model``."""
        state = torch.load(path, map_location="cpu", weights_only=True)
        model.load_state_dict(state)
        return model

    def run_inference(self, model: nn.Module, batch: Any) -> Any:
        """Forward pass; ``batch`` may be a tensor or ``(inputs, ...)`` tuple."""
        model.eval()
        inputs = batch[0] if isinstance(batch, (list, tuple)) else batch
        with torch.no_grad():
            return model(inputs)

    def to_device(self, model: nn.Module, device: str) -> nn.Module:
        """Move ``model`` to ``device``."""
        return model.to(device)
