# TensorPress

[![PyPI](https://img.shields.io/pypi/v/tensorpress.svg)](https://pypi.org/project/tensorpress/)
[![CI](https://img.shields.io/badge/ci-pending-lightgrey.svg)](#)
[![Coverage](https://img.shields.io/badge/coverage-pending-lightgrey.svg)](#)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

TensorPress compresses PyTorch convolutional neural networks by replacing
selected `Conv2d` layers with Tucker-2 or CPD tensor factorization modules. It
keeps the user workflow simple: configure compression intent, run the compressor,
inspect the result, optionally fine-tune, and export the compressed model.

## Installation

```bash
pip install tensorpress[torch]
```

For rich terminal reports:

```bash
pip install tensorpress[torch,rich]
```

## Minimal Example

```python
from tensorpress import CompressConfig, Compressor

model = MyModel()

cfg = CompressConfig(
    method="tucker",      # "tucker" or "cpd"
    layers="all",         # "all", names, regex, or callable
    ranks="auto",         # "auto", float target, or per-layer dict
    use_bn=False,
    finetune=False,
)

compressor = Compressor(cfg)
result = compressor.compress(model)

result.report()
result.export("compressed.pt")
```

With fine-tuning:

```python
from tensorpress.config import FinetuneConfig

cfg = CompressConfig(
    method="cpd",
    layers=lambda name, module: "features" in name,
    ranks="auto",
    finetune=True,
    finetune_config=FinetuneConfig(epochs=3, lr=1e-4),
)

result = Compressor(cfg).compress(
    model,
    dataloader={"train": train_loader, "val": val_loader},
)
```

## Documentation

The documentation source lives in [`docs/`](docs/). Build it with:

```bash
mkdocs build -f docs/mkdocs.yml
```

Runnable examples are available in [`examples/`](examples/):

```bash
python examples/quickstart.py --list-datasets
python examples/quickstart.py
python examples/quickstart.py --dataset fashion-mnist
python examples/cifar10_resnet18.py --no-pretrained --head-epochs 1 --ft-epochs 1
```

Dataset-backed tests are optional and skipped by default:

```bash
python -m pytest tests/test_datasets.py --run-dataset-tests --dataset-name cifar10
```

## Contributing

TensorPress is being migrated from research scripts into a library. Keep changes
small, typed, documented, and covered by tests. Run the test suite before opening
a pull request:

```bash
python -m pytest
```

## License

TensorPress is distributed under the license in [`LICENSE`](LICENSE).
