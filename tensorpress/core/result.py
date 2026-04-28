"""Compression run metadata and reporting."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger(__name__)


@dataclass
class CompressionResult:
    """Outcome of a ``Compressor.compress`` run.

    Parameters
    ----------
    model
        The (modified in-place) model after compression and optional finetuning.
    layer_stats
        Per-layer parameter counts and ranks from the replacement step.
    trainable_params_before
        Trainable parameter count before replacements.
    trainable_params_after
        Trainable parameter count after replacements.
    finetune_history
        Per-epoch training loss if finetuning ran; otherwise ``None``.
    backend
        Backend used for persistence and inference helpers.
    """

    model: Any
    layer_stats: dict[str, dict[str, Any]]
    trainable_params_before: int
    trainable_params_after: int
    finetune_history: list[float] | None = None
    backend: Any = field(repr=False, default=None)

    def report(self) -> None:
        """Print a human-readable before/after summary (Rich if installed)."""
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="TensorPress compression")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Trainable params (before)", str(self.trainable_params_before))
            table.add_row("Trainable params (after)", str(self.trainable_params_after))
            if self.layer_stats:
                ratio = self.trainable_params_after / max(self.trainable_params_before, 1)
                table.add_row("Overall param ratio (after/before)", f"{ratio:.4f}")
            Console().print(table)
        except ImportError:
            log.debug("rich not installed; plain-text report")
            print("TensorPress compression summary")
            print(f"  trainable params before: {self.trainable_params_before}")
            print(f"  trainable params after:  {self.trainable_params_after}")

    def compare(self) -> CompressionResult:
        """Print a short comparison and return ``self`` for chaining."""
        lines = [
            "TensorPress comparison",
            f"  layers touched: {len(self.layer_stats)}",
            f"  parameters: {self.trainable_params_before} -> {self.trainable_params_after}",
        ]
        if self.finetune_history:
            lines.append(f"  finetune epochs: {len(self.finetune_history)}")
        print("\n".join(lines))
        return self

    def export(self, path: str) -> None:
        """Save weights using the active backend (requires ``backend`` on this result)."""
        if self.backend is None:
            raise RuntimeError("CompressionResult has no backend; cannot export.")
        self.backend.save_model(self.model, path)
