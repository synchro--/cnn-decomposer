# AGENT_UTILS — Metrics, Profiling, Serialization

## Goal
Build supporting utilities for measuring, comparing, and persisting
compressed models.

---

## `tensorpress/utils/metrics.py`

```python
def count_parameters(model) -> int:
    """Count total trainable parameters."""
    ...

def count_flops(model, input_size: tuple) -> int:
    """
    Estimate FLOPs for a forward pass.
    Use torchinfo if available, fallback to manual Conv2d FLOPs formula.
    
    FLOPs for Conv2d = 2 * C_in * C_out * K_h * K_w * H_out * W_out
    """
    ...

def compression_ratio(original_params: int, compressed_params: int) -> float:
    ...

def parameter_reduction_pct(original: int, compressed: int) -> float:
    ...
```

---

## `tensorpress/utils/profiler.py`

```python
import time

class ModelProfiler:
    """
    Measures latency and throughput before/after compression.

    Usage
    -----
    with ModelProfiler(model, input_tensor) as p:
        # model runs here
    print(p.avg_latency_ms)
    """

    def __init__(self, model, input_tensor, n_warmup=5, n_runs=50):
        ...

    def profile(self) -> dict:
        """Returns dict: avg_ms, std_ms, throughput_fps."""
        ...

    @staticmethod
    def compare(original_result: dict, compressed_result: dict) -> dict:
        """Return speedup ratio and latency delta."""
        ...
```

---

## `tensorpress/utils/serialization.py`

```python
def save_compressed(model, config: "CompressConfig", path: str) -> None:
    """
    Save compressed model + config together so it can be fully restored.
    Format: torch checkpoint with 'state_dict' and 'compress_config' keys.
    """
    ...

def load_compressed(model_class, path: str):
    """
    Restore a compressed model from a checkpoint.
    Re-applies the compression structure before loading weights.
    """
    ...
```

---

## Output Contract
- `count_parameters`, `count_flops` return correct ints ✓
- `ModelProfiler.profile()` returns timing dict ✓
- `save_compressed` / `load_compressed` round-trip ✓
- `torchinfo` listed as optional dependency (not hard requirement) ✓
