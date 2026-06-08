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
python examples/cpd_all_conv.py --dataset fashion-mnist
python scripts/benchmark_native_cpd.py --datasets fashion-mnist cifar100
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

| Model | Total params | Conv params | FC params | Role in demos |
| --- | ---: | ---: | ---: | --- |
| `tinycnn` | 56k | 55k | 650 | Tucker vs CPD on a tiny all-conv-ish stack |
| `fashion-lenet-compact` | 2.71M | 2.52M | 198k | Tucker sweeps with a smaller FC head |
| `fashion-lenet` | 3.57M | 2.52M | 1.05M | Same conv stack, large FC — whole-model ratio dilution |
| `cpd-all-conv` | 52–99k | all | 0 | Native CP-factor architecture (trained from scratch) |

### Native CP architecture (`CpdAllConvNet`)

TensorPress can also **compress an existing** dense model post-hoc. A separate
demo trains a network where CP factorization is the architecture itself: every
logical conv is a four-stage chain (1×1 → k×1 depthwise → 1×k depthwise → 1×1),
and the classifier is conv-only — no `nn.Linear`. See
[`examples/cpd_all_conv.py`](examples/cpd_all_conv.py).

Compared to dense `FashionLeNet` (10k train / 4k test, 10 epochs, same
dataloaders):

| Dataset | Model | Params | Conv | FC | Accuracy |
| --- | --- | ---: | ---: | ---: | ---: |
| Fashion-MNIST | `fashion-lenet` | 3.57M | 2.51M | 1.05M | 89.6% |
| Fashion-MNIST | `cpd-all-conv` | **52k** | 52k | 0 | 83.6% |
| CIFAR-100 | `fashion-lenet` | 3.62M | 2.52M | 1.10M | 16.9% |
| CIFAR-100 | `cpd-all-conv` | **99k** | 99k | 0 | 17.6% |

On Fashion-MNIST the native CP net is ~**68×** smaller with a ~6 pp accuracy gap;
on CIFAR-100 it matches the large LeNet baseline at ~**37×** fewer parameters —
the tiny parameter budget also acts as implicit regularization, which helps on the
harder dataset. Raw rows: [`benchmark_native_cpd.json`](benchmark_native_cpd.json).

```bash
python examples/cpd_all_conv.py --dataset fashion-mnist --epochs 10
python scripts/benchmark_native_cpd.py --datasets fashion-mnist cifar100
```

### Case 1 — TinyCNN: Tucker vs CPD

Conv ≈ whole model (650-param classifier), so both methods shrink the network
end-to-end. CPD is the only LeNet-scale model where PARAFAC completes reliably.

| Ratio | Method | Baseline | Compressed | Δ pp | Conv × | Whole × |
| ---: | --- | ---: | ---: | ---: | ---: | ---: |
| 3 | Tucker | 75.3% | 74.9% | −0.4 | 2.91× | 2.85× |
| 8 | Tucker | 75.3% | 72.8% | −2.5 | 7.66× | 7.12× |
| 8 | **CPD** | 74.0% | 71.0% | −2.9 | 7.85× | 7.28× |
| 20 | Tucker | 75.3% | 68.0% | −7.3 | 18.51× | 15.40× |
| 20 | **CPD** | 74.0% | 69.0% | **−5.0** | 18.62× | 15.48× |
| 30 | Tucker | 75.3% | 63.4% | −11.9 | 25.56× | 19.92× |
| 30 | **CPD** | 74.0% | 67.8% | **−6.1** | 27.64× | **21.15×** |

At 20–30×, CPD keeps more accuracy than Tucker on TinyCNN while hitting similar
compression. Post-hoc CPD on FashionLeNet **OOM**s on conv3 PARAFAC — use Tucker
there, or train the native `cpd-all-conv` architecture instead.

```bash
python examples/quickstart.py --model tinycnn --method cpd --compression-ratio 20
```

### Case 2 — FashionLeNet compact: Tucker sweeps

~89% baseline, near-lossless up to ~3× conv compression; whole-model ratio tracks
conv more closely because the FC head is only **198k** params.

| Ratio | Baseline | Compressed | Δ pp | Conv × | Whole × | Params after |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| **3** | 89.2% | 87.5% | **−1.6** | 2.99× | 2.61× | 1,037,388 |
| 8 | 89.2% | 86.5% | −2.7 | 7.98× | 5.29× | 512,382 |
| 15 | 89.2% | 83.4% | −5.8 | 14.85× | 7.39× | 366,751 |
| 20 | 89.2% | 80.1% | −9.1 | 19.92× | 8.37× | 323,698 |
| 30 | 89.2% | 76.6% | −12.6 | 29.80× | **9.61×** | 281,919 |

```bash
python examples/quickstart.py \
  --model fashion-lenet-compact --dataset fashion-mnist \
  --epochs 10 --ft-epochs 5 \
  --train-size 10000 --test-size 4000 --compression-ratio 3.0
```

### Case 3 — Full vs compact FC: same conv compression, different whole-model ratio

Same conv stack (2.52M params); only the classifier size changes. At aggressive
ratios the **1.05M FC head** caps whole-model savings near **~3×** even when
convs compress **~30×**.

| Ratio | Model | Conv × | Whole × | Δ pp |
| ---: | --- | ---: | ---: | ---: |
| 8 | compact | 7.98× | **5.29×** | −2.7 |
| 8 | full FC | 7.98× | 2.60× | −2.6 |
| 20 | compact | 19.92× | **8.37×** | −9.1 |
| 20 | full FC | 19.92× | 3.02× | −9.6 |
| 30 | compact | 29.80× | **9.61×** | −12.6 |
| 30 | full FC | 29.80× | 3.13× | −13.8 |

`result.report()` separates conv-subset and whole-model scopes so this gap is
visible per run — not hidden behind a single headline number.

Reproduce full sweeps:

```bash
python scripts/sweep_models.py --model fashion-lenet-compact --method tucker
python scripts/sweep_models.py --model tinycnn --method cpd
python scripts/consolidate_sweeps.py
python scripts/print_sweep_tables.py   # print all ratios to stdout
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
