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
    requested_compression_ratio : float | None
        The N-fold compression target requested via ``compression_ratio``, used
        to report requested-vs-realized. ``None`` for ``"auto"`` or explicit
        manual ranks.

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
        requested_compression_ratio: float | None = None,
    ) -> None:
        self.model = model
        self.layer_stats = layer_stats
        self.trainable_params_before = trainable_params_before
        self.trainable_params_after = trainable_params_after
        self.finetune_history = finetune_history
        self.backend = backend
        self.requested_compression_ratio = requested_compression_ratio

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

    @property
    def compressed_params_before(self) -> int:
        """Parameter count of the compressed layers only, before compression."""
        return sum(int(s["original_params"]) for s in self.layer_stats.values())

    @property
    def compressed_params_after(self) -> int:
        """Parameter count of the compressed layers only, after compression."""
        return sum(int(s["compressed_params"]) for s in self.layer_stats.values())

    @property
    def subset_compression_ratio(self) -> float:
        """Realized N-fold compression ratio over the compressed layers only."""
        return self.compressed_params_before / max(self.compressed_params_after, 1)

    @property
    def untouched_params(self) -> int:
        """Trainable parameters in layers that were not compressed."""
        return max(self.trainable_params_before - self.compressed_params_before, 0)

    @staticmethod
    def _format_ranks(ranks: Any) -> str:
        """Render a stored rank value (int for CP, list for Tucker) compactly."""
        if isinstance(ranks, (list, tuple)):
            return "x".join(str(int(r)) for r in ranks)
        return str(int(ranks)) if isinstance(ranks, (int, float)) else str(ranks)

    def compare(self) -> dict[str, Any]:
        """Print a summary metrics block and return the metrics dict.

        The summary separates two scopes so an untouched classifier/BN cannot
        silently hide strong per-layer compression:

        - **compressed layers**: the layers actually factorized, and the N-fold
          compression realized there (versus the requested target, if any).
        - **whole model**: every trainable parameter, including untouched layers.

        Returns
        -------
        dict
            Metrics for both the compressed subset and the whole model.
        """
        summary: dict[str, Any] = {
            "compressed_subset": {
                "layers": len(self.layer_stats),
                "params_before": self.compressed_params_before,
                "params_after": self.compressed_params_after,
                "compression_ratio": round(self.subset_compression_ratio, 3),
                "requested_compression_ratio": self.requested_compression_ratio,
            },
            "whole_model": {
                "params_before": self.trainable_params_before,
                "params_after": self.trainable_params_after,
                "compression_ratio": round(self.compression_ratio, 3),
                "parameter_reduction_pct": round(self.parameter_reduction_pct, 1),
                "untouched_params": self.untouched_params,
            },
        }
        realized = f"{self.subset_compression_ratio:.2f}x"
        if self.requested_compression_ratio is not None:
            realized += f" (requested {self.requested_compression_ratio:.2f}x)"
        lines = [
            "TensorPress comparison",
            "  compressed layers:",
            f"    count:            {len(self.layer_stats)}",
            f"    params before:    {self.compressed_params_before:,}",
            f"    params after:     {self.compressed_params_after:,}",
            f"    compression:      {realized}",
            "  whole model (includes untouched layers):",
            f"    params before:    {self.trainable_params_before:,}",
            f"    params after:     {self.trainable_params_after:,}",
            f"    compression:      {self.compression_ratio:.2f}x",
            f"    untouched params: {self.untouched_params:,}",
        ]
        if self.finetune_history:
            lines.append(f"  finetune epochs:    {len(self.finetune_history)}")
        print("\n".join(lines))
        return summary

    def report(self) -> None:
        """Print a per-layer table with the chosen ranks, plus subset and
        whole-model summaries. Uses rich if installed, plain text otherwise.

        Three clearly-separated scopes are shown so that a large untouched layer
        (e.g. a dense classifier) does not mask strong per-layer compression:
        each compressed layer with its chosen rank, the compressed-subset
        aggregate (realized vs requested N-fold ratio), and the whole-model
        total with the count of untouched parameters.
        """
        realized = f"{self.subset_compression_ratio:.2f}x"
        if self.requested_compression_ratio is not None:
            realized += f" (req {self.requested_compression_ratio:.2f}x)"

        try:
            from rich.console import Console
            from rich.table import Table

            table = Table(title="TensorPress — Compression Report")
            table.add_column("Layer", style="cyan")
            table.add_column("Rank", justify="right", style="magenta")
            table.add_column("Before", justify="right")
            table.add_column("After", justify="right")
            table.add_column("Ratio", justify="right", style="green")
            for name, stats in self.layer_stats.items():
                table.add_row(
                    name,
                    self._format_ranks(stats.get("ranks")),
                    f"{stats['original_params']:,}",
                    f"{stats['compressed_params']:,}",
                    f"{stats['ratio']:.2f}x",
                )
            table.add_section()
            table.add_row(
                "[bold]Compressed layers[/bold]",
                "",
                f"[bold]{self.compressed_params_before:,}[/bold]",
                f"[bold]{self.compressed_params_after:,}[/bold]",
                f"[bold]{self.subset_compression_ratio:.2f}x[/bold]",
            )
            table.add_row(
                "[bold]Whole model[/bold]",
                "",
                f"[bold]{self.trainable_params_before:,}[/bold]",
                f"[bold]{self.trainable_params_after:,}[/bold]",
                f"[bold]{self.compression_ratio:.2f}x[/bold]",
            )
            console = Console()
            console.print(table)
            console.print(
                f"Compressed {len(self.layer_stats)} layer(s) by {realized}. "
                f"{self.untouched_params:,} param(s) in untouched layers "
                f"(e.g. classifier/BN) dilute the whole-model ratio."
            )
        except ImportError:
            print(f"\n{'Layer':<36} {'Rank':>10} {'Before':>10} {'After':>10} {'Ratio':>8}")
            print("-" * 78)
            for name, stats in self.layer_stats.items():
                print(
                    f"{name:<36} {self._format_ranks(stats.get('ranks')):>10} "
                    f"{stats['original_params']:>10,} "
                    f"{stats['compressed_params']:>10,} {stats['ratio']:>7.2f}x"
                )
            print("-" * 78)
            print(
                f"{'Compressed layers':<36} {'':>10} {self.compressed_params_before:>10,} "
                f"{self.compressed_params_after:>10,} {self.subset_compression_ratio:>7.2f}x"
            )
            print(
                f"{'Whole model':<36} {'':>10} {self.trainable_params_before:>10,} "
                f"{self.trainable_params_after:>10,} {self.compression_ratio:>7.2f}x"
            )
            print("-" * 78)
            print(
                f"Compressed {len(self.layer_stats)} layer(s) by {realized}. "
                f"{self.untouched_params:,} param(s) in untouched layers dilute the whole-model ratio."
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
