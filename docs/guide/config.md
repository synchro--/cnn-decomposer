# Configuration

`CompressConfig` is the main user-facing configuration object.

```python
from tensorpress import CompressConfig

cfg = CompressConfig(
    method="tucker",
    layers="all",
    ranks="auto",
    use_bn=False,
    finetune=False,
)
```

Configurations are dataclasses and can be serialized:

```python
payload = cfg.to_dict()
restored = CompressConfig.from_dict(payload)
```

Callable layer selectors can be serialized only as metadata because Python
callables cannot be reconstructed safely from JSON.

## Validation

`CompressConfig.validate()` raises `ValueError` for unsupported methods, invalid
layer specs, invalid rank specs, and non-boolean flags. `Compressor` calls
validation during initialization.
