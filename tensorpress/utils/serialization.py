"""Checkpoint helpers built on PyTorch (import inside functions where possible)."""

from __future__ import annotations

from typing import Any


def save_state_dict_to_file(model: Any, path: str) -> None:
    """Save ``model.state_dict()`` to ``path``."""
    import torch

    torch.save(model.state_dict(), path)


def load_state_dict_from_file(model: Any, path: str, *, map_location: str | None = None) -> Any:
    """Load a ``state_dict`` saved by :func:`save_state_dict_to_file` into ``model``."""
    import torch

    loc = map_location or "cpu"
    state = torch.load(path, map_location=loc, weights_only=True)
    model.load_state_dict(state)
    return model
