# Layer Selection

`LayerSelector` resolves `CompressConfig.layers` into concrete `(name, module)`
pairs.

## All Conv Layers

```python
CompressConfig(layers="all")
```

`None` is also treated as all convolution layers.

## Exact Names

```python
CompressConfig(layers=["features.0", "features.3"])
```

Use names from `model.named_modules()`.

## Regex

```python
CompressConfig(layers=r"features\.\d+")
```

Regex selection is useful for repeated block names.

## Callable Predicate

```python
import torch.nn as nn

cfg = CompressConfig(
    layers=lambda name, module: "layer" in name and isinstance(module, nn.Conv2d),
)
```

Predicates receive `(name, module)` and should return `True` for layers to
compress.
