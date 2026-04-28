"""Framework backends (PyTorch first)."""

from __future__ import annotations

from .base import BaseBackend
from .pytorch import PyTorchBackend


def get_backend(name: str) -> BaseBackend:
    """Return the backend implementation for ``name`` (currently only ``\"pytorch\"``)."""
    key = (name or "").lower().strip()
    if key == "pytorch":
        return PyTorchBackend()
    raise ValueError(f"Unsupported backend: {name!r}")


__all__ = ["BaseBackend", "PyTorchBackend", "get_backend"]
