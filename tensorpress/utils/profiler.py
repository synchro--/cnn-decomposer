"""Latency and throughput profiling for before/after compression comparison."""

from __future__ import annotations

import logging
import time
from typing import Any

import torch

log = logging.getLogger(__name__)


class ModelProfiler:
    """Measure inference latency and throughput on CPU or CUDA.

    Parameters
    ----------
    model : nn.Module
        Model to profile.
    input_tensor : torch.Tensor
        Representative input (batch already included).
    n_warmup : int, default=5
        Warm-up forward passes before timing starts (important for CUDA).
    n_runs : int, default=50
        Number of timed forward passes.

    Examples
    --------
    >>> p = ModelProfiler(model, torch.randn(1, 3, 32, 32))
    >>> stats = p.profile()
    >>> print(stats["avg_ms"], stats["throughput_fps"])
    """

    def __init__(
        self,
        model: Any,
        input_tensor: torch.Tensor,
        n_warmup: int = 5,
        n_runs: int = 50,
    ) -> None:
        self.model = model
        self.input_tensor = input_tensor
        self.n_warmup = n_warmup
        self.n_runs = n_runs
        self._use_cuda = input_tensor.device.type == "cuda"

    def profile(self) -> dict[str, float]:
        """Run timed forward passes and return latency statistics.

        Returns
        -------
        dict
            Keys: ``avg_ms``, ``std_ms``, ``min_ms``, ``max_ms``,
            ``throughput_fps``.
        """
        self.model.eval()
        x = self.input_tensor

        # Warm-up
        with torch.no_grad():
            for _ in range(self.n_warmup):
                self.model(x)

        if self._use_cuda:
            return self._profile_cuda(x)
        return self._profile_cpu(x)

    def _profile_cpu(self, x: torch.Tensor) -> dict[str, float]:
        times: list[float] = []
        with torch.no_grad():
            for _ in range(self.n_runs):
                t0 = time.perf_counter()
                self.model(x)
                times.append((time.perf_counter() - t0) * 1000.0)
        return self._stats(times, x.shape[0])

    def _profile_cuda(self, x: torch.Tensor) -> dict[str, float]:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        times: list[float] = []
        with torch.no_grad():
            for _ in range(self.n_runs):
                start.record()
                self.model(x)
                end.record()
                torch.cuda.synchronize()
                times.append(start.elapsed_time(end))
        return self._stats(times, x.shape[0])

    @staticmethod
    def _stats(times: list[float], batch_size: int) -> dict[str, float]:
        import statistics

        avg = statistics.mean(times)
        std = statistics.stdev(times) if len(times) > 1 else 0.0
        fps = (batch_size * 1000.0) / avg if avg > 0 else 0.0
        return {
            "avg_ms": round(avg, 3),
            "std_ms": round(std, 3),
            "min_ms": round(min(times), 3),
            "max_ms": round(max(times), 3),
            "throughput_fps": round(fps, 1),
        }

    @staticmethod
    def compare(before: dict[str, float], after: dict[str, float]) -> dict[str, float]:
        """Compute speedup and latency delta between two profile results.

        Parameters
        ----------
        before : dict
            Profile result from the original model.
        after : dict
            Profile result from the compressed model.

        Returns
        -------
        dict
            Keys: ``speedup_x``, ``latency_delta_ms``, ``fps_delta``.

        Examples
        --------
        >>> p_orig = ModelProfiler(original, x).profile()
        >>> p_comp = ModelProfiler(compressed, x).profile()
        >>> ModelProfiler.compare(p_orig, p_comp)
        {'speedup_x': 1.8, 'latency_delta_ms': -3.2, 'fps_delta': 120.5}
        """
        avg_before = before["avg_ms"]
        avg_after = after["avg_ms"]
        speedup = avg_before / avg_after if avg_after > 0 else 0.0
        return {
            "speedup_x": round(speedup, 2),
            "latency_delta_ms": round(avg_after - avg_before, 3),
            "fps_delta": round(after["throughput_fps"] - before["throughput_fps"], 1),
        }


def quick_compare(
    original: Any,
    compressed: Any,
    input_size: tuple[int, ...] = (1, 3, 32, 32),
    device: str | None = None,
    n_runs: int = 50,
) -> dict[str, Any]:
    """Profile both models and return a combined summary dict.

    Convenience wrapper around :class:`ModelProfiler` that handles device
    placement and returns a flat result ready to print or log.

    Parameters
    ----------
    original : nn.Module
    compressed : nn.Module
    input_size : tuple, default=(1, 3, 32, 32)
    device : str, optional
        Defaults to ``"cuda"`` if available, else ``"cpu"``.
    n_runs : int, default=50

    Returns
    -------
    dict
        Keys from ``before``, ``after``, and ``compare`` sub-dicts, plus
        ``device`` used.

    Examples
    --------
    >>> from tensorpress.utils.profiler import quick_compare
    >>> summary = quick_compare(model, result)  # CompressedModel proxies forward
    >>> print(summary)
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.randn(*input_size, device=device)
    orig = original.to(device)
    comp = compressed.to(device)

    before = ModelProfiler(orig, x, n_runs=n_runs).profile()
    after = ModelProfiler(comp, x, n_runs=n_runs).profile()
    delta = ModelProfiler.compare(before, after)

    log.info(
        "Profiling: %.2fms → %.2fms (%.2fx speedup)",
        before["avg_ms"], after["avg_ms"], delta["speedup_x"],
    )

    return {
        "device": device,
        "before": before,
        "after": after,
        "compare": delta,
    }
