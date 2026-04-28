"""Abstract backend for framework-specific operations."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Iterator


class BaseBackend(ABC):
    """Framework-specific helpers used by the compression pipeline."""

    @abstractmethod
    def get_conv_layers(self, model: Any) -> Iterator[tuple[str, Any]]:
        """Yield ``(name, module)`` for each 2D convolution layer in ``model``."""

    @abstractmethod
    def replace_module(self, model: Any, name: str, new_module: Any) -> None:
        """Replace a nested submodule called ``name`` with ``new_module`` (in-place)."""

    @abstractmethod
    def count_parameters(self, model: Any) -> int:
        """Return the number of trainable parameters in ``model``."""

    @abstractmethod
    def save_model(self, model: Any, path: str) -> None:
        """Persist model weights to ``path``."""

    @abstractmethod
    def load_model(self, model: Any, path: str) -> Any:
        """Load weights from ``path`` into ``model``."""

    @abstractmethod
    def run_inference(self, model: Any, batch: Any) -> Any:
        """Run a single forward pass with ``no_grad`` and return outputs."""

    @abstractmethod
    def to_device(self, model: Any, device: str) -> Any:
        """Move ``model`` to ``device`` (for example ``\"cpu\"``, ``\"cuda\"``, ``\"mps\"``)."""
