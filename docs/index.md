# TensorPress

TensorPress compresses PyTorch convolutional neural networks with tensor
factorization. It replaces selected `Conv2d` layers with Tucker-2 or CPD factor
layers, optionally fine-tunes the compressed model, and returns a reportable
`CompressionResult`.

## Five-Step Flow

```python
from tensorpress import CompressConfig, Compressor

model = MyModel()

cfg = CompressConfig(
    method="tucker",
    layers="all",
    ranks="auto",
    finetune=False,
)

compressor = Compressor(cfg)
result = compressor.compress(model)
result.report()
result.export("compressed.pt")
```

## What TensorPress Does

- Selects convolution layers by `"all"`, exact names, regex, or predicate.
- Estimates ranks automatically with VBMF or accepts manual per-layer ranks.
- Replaces each selected convolution with factorized PyTorch modules.
- Optionally runs a short pure-PyTorch fine-tuning loop.
- Saves the compressed model through the active backend.

Start with [Getting Started](getting_started.md), then see the guide pages for
configuration, layer selection, ranks, and fine-tuning.
