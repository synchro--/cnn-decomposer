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
(2,048 train / 1,024 test samples, 8 baseline epochs, 3 fine-tune epochs).
The `--ranks` float is a target parameter-retention fraction in `(0, 1]`; the
reported compression is the achieved parameter reduction.

| `--ranks` | Baseline acc | Compressed acc | Δ accuracy | Compression |
| --- | --- | --- | --- | --- |
| `0.2` | 60.4% | 56.8% | −3.6 pp | 8.7× |
| **`0.35`** | **60.5%** | **59.1%** | **−1.5 pp** | **9.8×** |
| `0.5` | 60.4% | 58.8% | −1.7 pp | 8.9× |
| `0.65` | 60.4% | 56.4% | −3.9 pp | 9.1× |
| **`1.0`** | **58.7%** | **55.6%** | **−3.1 pp** | **12.0×** |

Highlights:

- **Best accuracy-to-compression trade-off:** `--ranks 0.35` keeps **59.1%**
  accuracy (only **−1.5 pp**) while shrinking the convolutions **9.8×**.
- **Highest compression:** `--ranks 1.0` reaches **12.0×** at a modest
  **−3.1 pp** accuracy cost.

Reproduce the sweet spot with:

```bash
python examples/quickstart.py \
  --dataset fashion-mnist --epochs 8 --ft-epochs 3 --ranks 0.35
```

> Fine-tuning matters: without it (`--ft-epochs 0`) the same `--ranks 0.5`
> run drops from −1.7 pp to roughly −32 pp, so always fine-tune after
> aggressive compression.

## Contributing

TensorPress is being migrated from research scripts into a library. Keep changes
small, typed, documented, and covered by tests. Run the test suite before opening
a pull request:

```bash
python -m pytest
```

## License

TensorPress is distributed under the license in [`LICENSE`](LICENSE).
