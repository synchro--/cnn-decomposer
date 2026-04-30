# Getting Started

## Installation

```bash
pip install tensorpress[torch]
```

For richer terminal reports, install:

```bash
pip install tensorpress[torch,rich]
```

## Minimal Example

```python
from tensorpress import CompressConfig, Compressor

model = MyModel()

cfg = CompressConfig(
    method="tucker",
    layers="all",
    ranks="auto",
    use_bn=False,
    finetune=False,
)

result = Compressor(cfg).compress(model)
# result(x) forwards through the wrapped model; use result.model when you need nn.Module APIs.
result.report()
result.export("compressed.pt")
```

If you enable fine-tuning, pass a training dataloader:

```python
from tensorpress.config import FinetuneConfig

cfg = CompressConfig(
    method="cpd",
    layers=["features.0", "features.3"],
    ranks="auto",
    finetune=True,
    finetune_config=FinetuneConfig(epochs=3, lr=1e-4),
)

result = Compressor(cfg).compress(
    model,
    dataloader={"train": train_loader, "val": val_loader},
)
```

## Configuration Options

| Option | Values | Purpose |
| --- | --- | --- |
| `method` | `"tucker"`, `"cpd"` | Selects the decomposition algorithm. |
| `layers` | `None`, `"all"`, `list[str]`, regex `str`, callable | Chooses which conv layers to replace. |
| `ranks` | `"auto"`, `float`, `dict[str, Any]` | Chooses automatic, compression-targeted, or manual ranks. |
| `use_bn` | `bool` | Inserts `BatchNorm2d` after factor layers. |
| `finetune` | `bool` | Runs post-compression fine-tuning. |
| `finetune_config` | `FinetuneConfig` | Controls epochs, learning rate, scheduler, weight decay, and AMP. |

## Tucker vs CPD

Use Tucker first when you want a stable default for general convolution layers.
It replaces one convolution with three layers and usually gives predictable
compression.

Use CPD when you want a more aggressive factorization. It replaces one
convolution with four layers: pointwise, vertical depthwise, horizontal
depthwise, and pointwise.

## Saving And Loading

`CompressedModel.export()` writes model weights through the active backend:

```python
result.export("compressed.pt")
```

Load the saved weights with standard PyTorch APIs into the same compressed model
structure, or call `result(x)` / use `result.model` when you need the raw `nn.Module`.

## Example Datasets

The example scripts can run against several small, common torchvision
classification datasets:

```bash
python examples/quickstart.py --list-datasets
python examples/quickstart.py --dataset cifar10
python examples/quickstart.py --dataset fashion-mnist
```

Supported datasets are `cifar10`, `cifar100`, `mnist`, `fashion-mnist`,
`kmnist`, `svhn`, and `stl10`. Downloaded data lives under `data/`, which is
ignored by git.

Dataset-backed pytest checks are opt-in so normal tests never require a network
download:

```bash
python -m pytest tests/test_datasets.py --run-dataset-tests --dataset-name cifar10
```
