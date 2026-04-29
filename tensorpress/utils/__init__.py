"""Utilities: metrics, profiling, and serialization helpers."""

from .metrics import (
    compression_ratio,
    count_flops,
    count_parameters,
    parameter_reduction_pct,
    summarize,
)
from .profiler import ModelProfiler, quick_compare
from .serialization import load_state_dict_from_file, save_state_dict_to_file

__all__ = [
    # metrics
    "count_parameters",
    "count_flops",
    "compression_ratio",
    "parameter_reduction_pct",
    "summarize",
    # profiler
    "ModelProfiler",
    "quick_compare",
    # serialization
    "save_state_dict_to_file",
    "load_state_dict_from_file",
]
