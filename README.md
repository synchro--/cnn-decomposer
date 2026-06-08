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
python examples/quickstart.py --model fashion-lenet --dataset fashion-mnist
python examples/quickstart.py --model fashion-lenet-compact --method cpd --compression-ratio 20
python examples/cifar10_resnet18.py --no-pretrained --head-epochs 1 --ft-epochs 1
```

Dataset-backed tests are optional and skipped by default:

```bash
python -m pytest tests/test_datasets.py --run-dataset-tests --dataset-name cifar10
```

## Example Results

Fashion-MNIST sweeps (10,000 train / 4,000 test, 10 baseline epochs, 5
fine-tune epochs). `--compression-ratio` targets N-fold shrinkage of the **conv
stack**; **Conv ×** is the realized subset ratio, **Whole ×** includes the
untouched dense head. Full raw rows: [`sweep_results_consolidated.json`](sweep_results_consolidated.json).

| Model | Total params | Conv params | FC params |
| --- | ---: | ---: | ---: |
| `tinycnn` | 56k | 55k | 650 |
| `fashion-lenet` (full FC) | 3.57M | 2.52M | 1.05M |
| `fashion-lenet-compact` | 2.71M | 2.52M | 198k |

### Tucker (all three models)

| Model | Ratio | Baseline | Compressed | Δ pp | Conv × | Whole × | Reduction % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tinycnn | 3 | 75.3% | 74.9% | −0.4 | 2.91× | 2.85× | 64.9% |
| tinycnn | 8 | 75.3% | 72.8% | −2.5 | 7.66× | 7.12× | 85.9% |
| tinycnn | 20 | 75.3% | 68.0% | −7.3 | 18.51× | 15.40× | 93.5% |
| tinycnn | 30 | 75.3% | 63.4% | −11.9 | 25.56× | 19.92× | 95.0% |
| full | 3 | 89.8% | 88.5% | −1.3 | 2.99× | 1.88× | 46.9% |
| full | 8 | 89.4% | 86.8% | −2.6 | 7.98× | 2.60× | 61.6% |
| full | 20 | 89.3% | 79.8% | −9.6 | 19.92× | 3.02× | 66.9% |
| full | 30 | 89.7% | 75.8% | −13.8 | 29.80× | 3.13× | 68.1% |
| compact | 3 | 89.2% | 87.5% | −1.6 | 2.99× | 2.61× | 61.7% |
| compact | 8 | 89.2% | 86.5% | −2.7 | 7.98× | 5.29× | 81.1% |
| compact | 20 | 89.2% | 80.1% | −9.1 | 19.92× | 8.37× | 88.0% |
| compact | 30 | 89.2% | 76.6% | −12.6 | 29.80× | 9.61× | 89.6% |

### CPD (TinyCNN; FashionLeNet variants OOM on conv3 PARAFAC)

| Model | Ratio | Baseline | Compressed | Δ pp | Conv × | Whole × | Reduction % |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| tinycnn | 8 | 74.0% | 71.0% | −2.9 | 7.85× | 7.28× | 86.3% |
| tinycnn | 15 | 74.0% | 69.7% | −4.3 | 14.54× | 12.57× | 92.0% |
| tinycnn | 20 | 74.0% | 69.0% | −5.0 | 18.62× | 15.48× | 93.5% |
| tinycnn | 30 | 74.0% | 67.8% | −6.1 | 27.64× | 21.15× | 95.3% |

**Takeaways**

- **`fashion-lenet-compact`** is the best demo for *whole-model* compression:
  at 20× conv target the model shrinks **8.4×** overall (vs **3.0×** for the
  full FC head) because the dense classifier is only **198k** params.
- **CPD** works end-to-end on **`tinycnn`** (max conv rank ~27 at 30×); on
  FashionLeNet's **conv3** (384×192×5×5) PARAFAC at rank ≥157 OOMs on CPU —
  Tucker is the practical choice for the wide LeNet stack.
- **Accuracy cliff near 20×:** all LeNet variants lose **~9–14 pp** at 20–30×
  conv compression despite fine-tuning; TinyCNN is more robust (conv ≈ whole model).

Reproduce sweeps:

```bash
python scripts/sweep_models.py --model fashion-lenet-compact --method tucker
python scripts/sweep_models.py --model tinycnn --method cpd
python scripts/consolidate_sweeps.py
```

Sweet spot (Tucker, compact):

```bash
python examples/quickstart.py \
  --model fashion-lenet-compact --dataset fashion-mnist \
  --epochs 10 --ft-epochs 5 \
  --train-size 10000 --test-size 4000 --compression-ratio 3.0
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
