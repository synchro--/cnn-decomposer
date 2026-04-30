"""Compressed model wrapper: metrics, reporting, and transparent PyTorch proxies."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


class CompressedModel:
    """
    Output of ``Compressor.compress()``.

    Acts as a thin wrapper around the compressed ``nn.Module``. Common
    PyTorch operations (forward pass, eval/train mode, device transfer)
    are proxied directly so the wrapper is largely transparent.

    Parameters
    ----------
    model : nn.Module
        The in-place compressed model.
    layer_stats : dict
        Per-layer dict with original_params, compressed_params, ranks, ratio.
    trainable_params_before : int
        Trainable parameter count before replacement.
    trainable_params_after : int
        Trainable parameter count after replacement.
    finetune_history : list[float] | None
        Per-epoch training loss if finetune=True, else None.
    backend : BaseBackend | None
        Used by ``export()``; may be None in tests.

    Examples
    --------
    >>> result = Compressor(cfg).compress(model)  # doctest: +SKIP
    >>> result(x)  # forward pass — no .model needed  # doctest: +SKIP
    >>> result.eval()  # doctest: +SKIP
    >>> result.to("cuda")  # doctest: +SKIP
    >>> result.report()  # doctest: +SKIP
    >>> result.export("out.pt")  # doctest: +SKIP
    """

    def __init__(
        self,
        model: Any,
        layer_stats: dict[str, dict[str, Any]],
        trainable_params_before: int,
        trainable_params_after: int,
        finetune_history: list[float] | None = None,
        backend: Any = None,
    ) -> None:
        self.model = model
        self.layer_stats = layer_stats
        self.trainable_params_before = trainable_params_before
        self.trainable_params_after = trainable_params_after
        self.finetune_history = finetune_history
        self.backend = backend

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        """Forward pass — delegates to the underlying model."""
        return self.model(*args, **kwargs)

    def eval(self) -> CompressedModel:
        """Set model to eval mode and return ``self`` for chaining."""
        self.model.eval()
        return self

    def train(self, mode: bool = True) -> CompressedModel:
        """Set model training mode and return ``self`` for chaining."""
        self.model.train(mode)
        return self

    def to(self, *args: Any, **kwargs: Any) -> CompressedModel:
        """Move model to device/dtype and return ``self`` for chaining."""
        self.model.to(*args, **kwargs)
        return self

    def parameters(self, recurse: bool = True):
        """Proxy to ``model.parameters()``."""
        return self.model.parameters(recurse=recurse)

    def state_dict(self, **kwargs: Any) -> dict[str, Any]:
        """Proxy to ``model.state_dict()``."""
        return self.model.state_dict(**kwargs)

    @property
    def compression_ratio(self) -> float:
        """Original / compressed parameter count."""
        return self.trainable_params_before / max(self.trainable_params_after, 1)

    @property
    def parameter_reduction_pct(self) -> float:
        """Percentage of parameters removed (0–100)."""
        return 100.0 * (
            1.0 - self.trainable_params_after / max(self.trainable_params_before, 1)
        )

    def compare(self) -> dict[str, Any]:
        """Print a summary metrics block and return the metrics dict.

        Returns
        -------
        dict
            Keys: ``original_params``, ``compressed_params``,
            ``compression_ratio``, ``parameter_reduction_pct``.
        """
        summary = {
            "original_params": self.trainable_params_before,
            "compressed_params": self.trainable_params_after,
            "compression_ratio": round(self.compression_ratio, 3),
            "parameter_reduction_pct": round(self.parameter_reduction_pct, 1),
        }
        lines = [
            "TensorPress comparison",
            f"  layers touched:     {len(self.layer_stats)}",
            f"  params before:      {self.trainable_params_before:,}",
            f"  params after:       {self.trainable_params_after:,}",
            f"  compression ratio:  {self.compression_ratio:.2f}x",
            f"  reduction:          {self.parameter_reduction_pct:.1f}%",
        ]
        if self.finetune_history:
            lines.append(f"  finetune epochs:    {len(self.finetune_history)}")
        print("\n".join(lines))
        return summary

    def report(self) -> None:
        """Print a per-layer table. Uses rich if installed, plain text otherwise."""
        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="TensorPress — Compression Report")
            table.add_column("Layer", style="cyan")
            table.add_column("Before", justify="right")
            table.add_column("After", justify="right")
            table.add_column("Ratio", justify="right", style="green")
            for name, stats in self.layer_stats.items():
                table.add_row(
                    name,
                    f"{stats['original_params']:,}",
                    f"{stats['compressed_params']:,}",
                    f"{stats['ratio']:.2f}x",
                )
            table.add_row(
                "[bold]TOTAL[/bold]",
                f"[bold]{self.trainable_params_before:,}[/bold]",
                f"[bold]{self.trainable_params_after:,}[/bold]",
                f"[bold]{self.compression_ratio:.2f}x[/bold]",
            )
            Console().print(table)
        except ImportError:
            print(f"\n{'Layer':<40} {'Before':>10} {'After':>10} {'Ratio':>8}")
            print("-" * 70)
            for name, stats in self.layer_stats.items():
                print(
                    f"{name:<40} {stats['original_params']:>10,} "
                    f"{stats['compressed_params']:>10,} {stats['ratio']:>7.2f}x"
                )
            print("-" * 70)
            print(
                f"{'TOTAL':<40} {self.trainable_params_before:>10,} "
                f"{self.trainable_params_after:>10,} {self.compression_ratio:>7.2f}x"
            )

    def export(self, path: str) -> None:
        """Save compressed model weights to ``path``."""
        if self.backend is None:
            raise RuntimeError("CompressedModel has no backend; cannot export.")
        self.backend.save_model(self.model, path)
        log.info("Saved compressed model to %s", path)

    def __repr__(self) -> str:
        return (
            f"<CompressedModel "
            f"ratio={self.compression_ratio:.2f}x "
            f"reduction={self.parameter_reduction_pct:.1f}%>"
        )
