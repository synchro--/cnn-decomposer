# AGENT_BACKEND — Framework Adapters

## Goal
Isolate every framework-specific operation behind a `BaseBackend` ABC.
The PyTorch implementation wraps the utility functions already present in
`pytorch_utils.py` (after AGENT_PYTORCH_FIXES has cleaned them up).

---

## Key insight from reading `pytorch_utils.py`

These utility functions already exist and should be WRAPPED (not rewritten):
- `set_layer_weights(layer, tensor)` — assigns numpy weight array to a layer
- `get_layer_weights(layer)` — retrieves weights as numpy
- `save_checkpoint(state, is_best, checkpoint)` — saves model state dict
- `load_checkpoint(checkpoint, model, optimizer)` — loads state dict

Do NOT copy them verbatim. Import and delegate to them from `PyTorchBackend`.

---

## Step 1: `tensorpress/backends/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any, Iterator

class BaseBackend(ABC):
    """
    Abstract interface for framework-specific operations.
    Implementations exist for PyTorch (v1), TensorFlow and JAX (roadmap).
    """

    @abstractmethod
    def get_conv_layers(self, model: Any) -> Iterator[tuple[str, Any]]:
        """Yield (dotted_name, layer) for every Conv2d in the model."""
        ...

    @abstractmethod
    def replace_module(self, model: Any, name: str, new_module: Any) -> None:
        """
        Replace a submodule identified by dotted name in-place.
        Must handle nested names like 'layer1.0.conv1'.
        """
        ...

    @abstractmethod
    def count_parameters(self, model: Any) -> int:
        """Return total number of trainable parameters."""
        ...

    @abstractmethod
    def save_model(self, model: Any, path: str) -> None:
        """Persist model weights to disk."""
        ...

    @abstractmethod
    def load_model(self, model: Any, path: str) -> Any:
        """Load weights from disk into model."""
        ...

    @abstractmethod
    def run_inference(self, model: Any, batch: Any) -> Any:
        """Single forward pass under no_grad context. Returns output."""
        ...

    @abstractmethod
    def to_device(self, model: Any, device: str) -> Any:
        """Move model to specified device ('cpu', 'cuda', 'mps')."""
        ...
```

---

## Step 2: `tensorpress/backends/pytorch.py`

```python
import torch
import torch.nn as nn
from .base import BaseBackend

class PyTorchBackend(BaseBackend):

    def get_conv_layers(self, model):
        """Yield (name, module) for all Conv2d layers using named_modules()."""
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d):
                yield name, module

    def replace_module(self, model, name, new_module):
        """
        Navigate dotted path and setattr on parent.
        E.g. 'layer1.0.conv1' → getattr(model, 'layer1')[0], setattr(..., 'conv1', new_module)
        """
        parts = name.split('.')
        parent = model
        for part in parts[:-1]:
            # Handle both attribute access and integer indexing (for Sequential)
            if part.isdigit():
                parent = parent[int(part)]
            else:
                parent = getattr(parent, part)
        last = parts[-1]
        if last.isdigit():
            parent[int(last)] = new_module
        else:
            setattr(parent, last, new_module)

    def count_parameters(self, model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def save_model(self, model, path):
        torch.save(model.state_dict(), path)

    def load_model(self, model, path):
        model.load_state_dict(torch.load(path, map_location='cpu'))
        return model

    def run_inference(self, model, batch):
        device = next(model.parameters()).device
        with torch.no_grad():
            if isinstance(batch, (list, tuple)):
                inputs = batch[0].to(device)
            else:
                inputs = batch.to(device)
            return model(inputs)

    def to_device(self, model, device):
        return model.to(device)
```

---

## Step 3: Backend Registry `tensorpress/backends/__init__.py`

```python
from .base import BaseBackend
from .pytorch import PyTorchBackend

_REGISTRY: dict[str, type[BaseBackend]] = {
    "pytorch": PyTorchBackend,
}

def get_backend(name: str) -> BaseBackend:
    if name not in _REGISTRY:
        raise ValueError(
            f"Backend '{name}' not available. "
            f"Supported: {list(_REGISTRY)}"
        )
    return _REGISTRY[name]()
```

---

## Step 4: Stub files for future frameworks

Create `tensorpress/backends/tensorflow.py`:
```python
from .base import BaseBackend

class TensorFlowBackend(BaseBackend):
    def get_conv_layers(self, model): raise NotImplementedError("TF backend: v0.3")
    # ... same for all other methods
```

Same pattern for `tensorpress/backends/jax.py`.
Do NOT register them in `_REGISTRY` yet — uncomment when implemented.

---

## Critical: `replace_module` correctness

Test this specifically — it must handle:
- `"conv1"` → top-level attribute
- `"features.0"` → Sequential with integer index
- `"layer1.0.conv1"` → ResNet-style nested modules

```python
# Verify with:
import torchvision.models as models
model = models.resnet18()
backend = PyTorchBackend()
names = [name for name, _ in backend.get_conv_layers(model)]
# e.g. 'layer1.0.conv1', 'layer1.0.conv2', etc.
backend.replace_module(model, names[0], nn.Identity())
assert isinstance(model.layer1[0].conv1, nn.Identity)
```

---

## Output Contract
- `get_backend("pytorch")` returns a working `PyTorchBackend` instance ✓
- `replace_module` correctly handles all dotted-name patterns ✓
- `save_model` / `load_model` round-trip restores identical state_dict ✓
- TF and JAX stubs exist with `NotImplementedError` ✓
