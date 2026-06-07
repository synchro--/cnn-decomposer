"""Timestamped JSON output helpers for TensorPress examples."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from tensorpress.core.result import CompressedModel


def make_results_path(
    prefix: str,
    *,
    output_dir: str | Path = "outputs",
    extension: str = "json",
) -> Path:
    """Return a unique timestamped output path and ensure the directory exists.

    Parameters
    ----------
    prefix : str
        Short run label, for example ``"quickstart"`` or ``"resnet18"``.
    output_dir : str | Path, default="outputs"
        Directory where result dumps are written.
    extension : str, default="json"
        File extension without a leading dot.

    Returns
    -------
    Path
        Path such as ``outputs/quickstart_20260606T153045.json``.
    """
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return directory / f"{prefix}_{stamp}.{extension}"


def save_run_results(
    path: str | Path,
    *,
    result: CompressedModel,
    run: dict[str, Any],
    metrics: dict[str, Any] | None = None,
    export_path: str | Path | None = None,
) -> Path:
    """Write a JSON dump of run configuration and compression results.

    Parameters
    ----------
    path : str | Path
        Destination JSON file.
    result : CompressedModel
        Output from :meth:`Compressor.compress`.
    run : dict[str, Any]
        Run metadata such as CLI arguments and config values.
    metrics : dict[str, Any] | None, default=None
        Extra evaluation metrics, for example accuracy before/after.
    export_path : str | Path | None, default=None
        Path to the exported model weights, if saved.

    Returns
    -------
    Path
        Written file path.
    """
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "run": run,
        "metrics": metrics or {},
        "compression": {
            "whole_model": {
                "trainable_params_before": result.trainable_params_before,
                "trainable_params_after": result.trainable_params_after,
                "compression_ratio": round(result.compression_ratio, 3),
                "parameter_reduction_pct": round(result.parameter_reduction_pct, 1),
                "untouched_params": result.untouched_params,
            },
            "compressed_subset": {
                "layers_touched": len(result.layer_stats),
                "params_before": result.compressed_params_before,
                "params_after": result.compressed_params_after,
                "compression_ratio": round(result.subset_compression_ratio, 3),
                "realized_keep_fraction": round(result.realized_keep_fraction, 4),
                "requested_keep_fraction": result.requested_keep_fraction,
            },
        },
        "layer_stats": result.layer_stats,
        "finetune_history": result.finetune_history,
    }
    if export_path is not None:
        payload["export_path"] = str(export_path)

    with destination.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, default=str)
        handle.write("\n")

    return destination
