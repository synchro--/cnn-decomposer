"""Parameter counting, FLOPs estimation, and compression metrics."""

from __future__ import annotations

import logging
from typing import Any

log = logging.getLogger(__name__)


def count_parameters(model: Any, trainable_only: bool = True) -> int:
    """Count parameters in a model.

    Parameters
    ----------
    model : nn.Module
        Model to inspect.
    trainable_only : bool, default=True
        If True, count only parameters where ``requires_grad=True``.

    Returns
    -------
    int
        Total parameter count.
    """
    return sum(
        p.numel()
        for p in model.parameters()
        if (p.requires_grad if trainable_only else True)
    )


def count_flops(model: Any, input_size: tuple[int, ...]) -> int:
    """Estimate FLOPs for a single forward pass.

    Uses ``torchinfo`` when available for an accurate traversal of the
    compute graph. Falls back to a manual ``Conv2d``-only formula when
    ``torchinfo`` is not installed (``pip install tensorpress[flops]``).

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    input_size : tuple[int, ...]
        Input tensor shape including batch dimension, e.g. ``(1, 3, 32, 32)``.

    Returns
    -------
    int
        Estimated total multiply-accumulate operations (MACs × 2 ≈ FLOPs).
    """
    try:
        from torchinfo import summary  # type: ignore[import]

        stats = summary(model, input_size=input_size, verbose=0)
        return int(stats.total_mult_adds * 2)
    except ImportError:
        log.debug("torchinfo not installed; using Conv2d-only FLOPs estimate")
        return _manual_conv_flops(model, input_size)


def _manual_conv_flops(model: Any, input_size: tuple[int, ...]) -> int:
    """Estimate FLOPs by walking Conv2d layers with a tracked spatial size.

    Formula per layer:
        FLOPs = 2 × C_in × C_out × K_h × K_w × H_out × W_out

    This is an approximation — it ignores non-Conv2d layers and assumes
    square spatial dimensions shrink only via stride.
    """
    import torch.nn as nn

    if len(input_size) != 4:
        raise ValueError("input_size must be (N, C, H, W)")

    _, c_in, h, w = input_size
    total = 0

    for module in model.modules():
        if not isinstance(module, nn.Conv2d):
            continue
        k_h, k_w = module.kernel_size if isinstance(module.kernel_size, tuple) else (module.kernel_size, module.kernel_size)
        s_h, s_w = module.stride if isinstance(module.stride, tuple) else (module.stride, module.stride)
        p_h, p_w = module.padding if isinstance(module.padding, tuple) else (module.padding, module.padding)

        h_out = (h + 2 * p_h - k_h) // s_h + 1
        w_out = (w + 2 * p_w - k_w) // s_w + 1

        flops = 2 * module.in_channels * module.out_channels * k_h * k_w * h_out * w_out
        total += flops

        # Update spatial dims for next layer (rough approximation)
        h, w = h_out, w_out

    return total


def compression_ratio(original: int, compressed: int) -> float:
    """Return ``original / compressed`` (how many times smaller).

    Parameters
    ----------
    original : int
        Parameter count before compression.
    compressed : int
        Parameter count after compression.

    Returns
    -------
    float
        Compression ratio ≥ 1.0. Returns 1.0 if compressed >= original.
    """
    if compressed <= 0:
        return float(original)
    return original / compressed


def parameter_reduction_pct(original: int, compressed: int) -> float:
    """Return percentage of parameters removed (0–100).

    Parameters
    ----------
    original : int
    compressed : int

    Returns
    -------
    float
        Percentage reduction, e.g. 60.0 means 60% fewer parameters.
    """
    if original <= 0:
        return 0.0
    return 100.0 * (1.0 - compressed / original)


def summarize(original: int, compressed: int) -> dict[str, Any]:
    """Return a dict of all compression metrics at once.

    Parameters
    ----------
    original : int
        Trainable parameter count before compression.
    compressed : int
        Trainable parameter count after compression.

    Returns
    -------
    dict
        Keys: ``original_params``, ``compressed_params``,
        ``compression_ratio``, ``parameter_reduction_pct``.
    """
    return {
        "original_params": original,
        "compressed_params": compressed,
        "compression_ratio": round(compression_ratio(original, compressed), 3),
        "parameter_reduction_pct": round(parameter_reduction_pct(original, compressed), 1),
    }
