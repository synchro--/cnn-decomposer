# AGENT_CONFIG — Config Schema & Layer Selection

## Goal
Create a fully typed, serializable configuration layer that the user
will interact with directly. This is the user-facing API surface for
declaring compression intent.

## Steps

### 1. `tensorpress/config/schema.py`

```python
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Literal
import json

@dataclass
class FinetuneConfig:
    epochs: int = 5
    lr: float = 1e-4
    scheduler: str | None = "cosine"   # "cosine" | "step" | None
    warmup_steps: int = 0

    def to_json(self) -> str:
        return json.dumps(asdict(self))


@dataclass
class CompressConfig:
    method: Literal["tucker", "cpd"] = "tucker"

    # Layer selection:
    # "all"            → every Conv2d
    # list[str]        → named layers (exact match)
    # str (regex)      → regex match on layer names
    # callable         → predicate(name, module) → bool
    layers: str | list[str] | Callable = "all"

    # Rank specification:
    # "auto"           → VBMF heuristic per layer
    # float (0–1)      → compression ratio (e.g. 0.5 = half the params)
    # dict[str, ...]   → per-layer manual ranks
    ranks: str | float | dict = "auto"

    finetune: bool = False
    finetune_config: FinetuneConfig = field(default_factory=FinetuneConfig)

    # Internal — set by Compressor, not user
    _framework: str = field(default="pytorch", init=False, repr=False)

    def to_dict(self) -> dict[str, Any]:
        ...  # serialize, handle callable layers specially

    @classmethod
    def from_dict(cls, d: dict) -> "CompressConfig":
        ...

    def validate(self) -> None:
        """Raise ValueError on invalid combinations."""
        ...
```

### 2. `tensorpress/config/selectors.py`

Build `LayerSelector` that resolves a `CompressConfig.layers` value
into a concrete list of `(name, module)` pairs from a model.

```python
class LayerSelector:
    def __init__(self, spec):
        self.spec = spec

    def select(self, model) -> list[tuple[str, nn.Module]]:
        """Return (name, module) pairs matching the spec."""
        if self.spec == "all":
            return self._all_conv(model)
        elif isinstance(self.spec, list):
            return self._by_names(model, self.spec)
        elif isinstance(self.spec, str):
            return self._by_regex(model, self.spec)
        elif callable(self.spec):
            return self._by_predicate(model, self.spec)
        raise ValueError(f"Unknown layer spec: {self.spec!r}")
```

### 3. `tensorpress/config/rank_estimators.py`

```python
class RankEstimator:
    def __init__(self, spec):
        self.spec = spec   # "auto" | float | dict

    def estimate(self, name: str, layer) -> tuple:
        if self.spec == "auto":
            return VBMFEstimator().estimate(layer)
        elif isinstance(self.spec, float):
            return self._from_ratio(layer, self.spec)
        elif isinstance(self.spec, dict):
            return self.spec[name]
        raise ValueError(...)


class VBMFEstimator:
    """
    Variational Bayesian Matrix Factorization rank estimator.
    Ported from original research code.
    """
    def estimate(self, layer) -> tuple:
        ...  # existing VBMF logic goes here
```

### 4. `tensorpress/config/__init__.py`
```python
from .schema import CompressConfig, FinetuneConfig
from .selectors import LayerSelector
from .rank_estimators import RankEstimator, VBMFEstimator
```

## Output Contract
- `tensorpress/config/schema.py` ✓ — CompressConfig, FinetuneConfig
- `tensorpress/config/selectors.py` ✓ — LayerSelector
- `tensorpress/config/rank_estimators.py` ✓ — RankEstimator, VBMFEstimator
- All dataclasses JSON round-trip ✓
- `CompressConfig.validate()` raises on bad input ✓
