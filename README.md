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

Install into an isolated environment. Quote the extras so your shell does not
expand the brackets (required in zsh):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install 'tensorpress[torch]'
```

For rich terminal reports:

```bash
pip install 'tensorpress[torch,rich]'
```

Using [uv](https://docs.astral.sh/uv/):

```bash
uv venv
uv pip install 'tensorpress[torch]'
```

See [Getting Started](docs/getting_started.md) for contributor setup with the
committed lockfile (`uv sync`).

## Minimal Example

```python
from tensorpress import CompressConfig, Compressor

model = MyModel()

cfg = CompressConfig(
    method="tucker",        # "tucker" or "cpd"
    layers="all",           # "all", names, regex, or callable
    compression_ratio=4.0,  # target N-fold size reduction (>= 1); ~4x smaller
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

## Example Results

Tucker-2 compression of the quickstart `TinyCNN` on **Fashion-MNIST**
(10,000 train / 4,000 test samples, 10 baseline epochs, 5 fine-tune epochs).
`--compression-ratio` is the requested N-fold size reduction for the compressed
conv layers; the "Compression" column is the achieved reduction on those layers.

| `--compression-ratio` | Baseline acc | Compressed acc | Δ accuracy | Compression |
| --- | --- | --- | --- | --- |
| `1.5` | 74.0% | 74.3% | +0.4 pp | 1.48× |
| **`2.0`** | **74.0%** | **73.8%** | **−0.2 pp** | **1.95×** |
| **`3.0`** | **74.6%** | **74.2%** | **−0.4 pp** | **2.85×** |
| `4.0` | 73.2% | 70.8% | −2.4 pp | 3.81× |
| `6.0` | 73.7% | 67.3% | −6.4 pp | 5.61× |
| **`8.0`** | **74.0%** | **71.0%** | **−3.0 pp** | **7.12×** |

Highlights:

- **Near-lossless up to ~3×:** `--compression-ratio 3.0` shrinks the
  convolutions **2.85×** while keeping accuracy within **−0.4 pp** of baseline.
- **Aggressive compression:** `--compression-ratio 8.0` reaches **7.12×** for a
  modest **−3.0 pp**, after fine-tuning recovers the initial drop.

Reproduce the sweet spot with:

```bash
python examples/quickstart.py \
  --dataset fashion-mnist --epochs 10 --ft-epochs 5 \
  --train-size 10000 --test-size 4000 --compression-ratio 3.0
```

> Fine-tuning matters: aggressive ratios drop sharply right after factorization
> and recover most of the gap over a few fine-tune epochs, so always fine-tune
> after compressing.

## Contributing

TensorPress is being migrated from research scripts into a library. Keep changes
small, typed, documented, and covered by tests. Run the test suite before opening
a pull request:

```bash
python -m pytest
```

## License

TensorPress is distributed under the license in [`LICENSE`](LICENSE).
