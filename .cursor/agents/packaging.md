# AGENT_PACKAGING — PyPI-Ready Package

## Goal
Set up everything needed to `pip install tensorpress` from PyPI
and run CI/CD via GitHub Actions.

**Version baseline (verified Feb 2026):**
- Python: ≥ 3.10 (PyTorch 2.10 dropped 3.9)
- PyTorch: ≥ 2.10.0
- PyTorch Lightning / Fabric: ≥ 2.6.1 (optional)
- torchinfo: ≥ 1.8.0
- tensorly: ≥ 0.8.1
- rich: ≥ 13.9
- numpy: ≥ 2.0
- scipy: ≥ 1.14
- ruff: ≥ 0.9
- pytest: ≥ 8.3
- mypy: ≥ 1.13

---

## `pyproject.toml`

```toml
[build-system]
requires = ["hatchling>=1.27"]
build-backend = "hatchling.build"

[project]
name = "tensorpress"
version = "0.1.0"
description = "Tensor factorization compression (Tucker & CPD) for neural network conv layers"
readme = "README.md"
license = { file = "LICENSE" }
requires-python = ">=3.10"
authors = [{ name = "Your Name", email = "you@example.com" }]
keywords = ["deep learning", "model compression", "tucker", "cpd", "tensor decomposition", "pytorch"]
classifiers = [
    "Development Status :: 3 - Alpha",
    "Intended Audience :: Science/Research",
    "Topic :: Scientific/Engineering :: Artificial Intelligence",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Programming Language :: Python :: 3.12",
    "Programming Language :: Python :: 3.13",
]

# Core deps — keep minimal. torch is NOT listed here because users
# should install it themselves with their preferred CUDA version.
# We declare it as a soft requirement via [project.optional-dependencies].
dependencies = [
    "tensorly>=0.8.1",
    "numpy>=2.0",
    "scipy>=1.14",        # used by VBMF SVD routines
]

[project.optional-dependencies]
# torch is intentionally optional — users choose their own CUDA build
torch   = ["torch>=2.10.0", "torchvision>=0.25.0"]
rich    = ["rich>=13.9"]           # pretty result.report() table
flops   = ["torchinfo>=1.8.0"]    # FLOPs counting in metrics

# Developer toolchain
dev = [
    "torch>=2.10.0",
    "torchvision>=0.25.0",
    "pytest>=8.3",
    "pytest-cov>=6.0",
    "ruff>=0.9",
    "mypy>=1.13",
    "rich>=13.9",
    "torchinfo>=1.8.0",
]

# Convenience alias
all = ["tensorpress[torch,rich,flops]"]

[project.urls]
Homepage      = "https://github.com/yourname/tensorpress"
Documentation = "https://yourname.github.io/tensorpress"
Changelog     = "https://github.com/yourname/tensorpress/blob/main/CHANGELOG.md"

[tool.ruff]
target-version = "py310"
line-length = 99

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]
ignore = ["E501"]   # line length handled by formatter

[tool.ruff.lint.isort]
known-first-party = ["tensorpress"]

[tool.mypy]
python_version = "3.10"
strict = false
ignore_missing_imports = true
warn_unused_ignores = true

[tool.pytest.ini_options]
testpaths = ["tests"]
markers   = ["integration: marks slow end-to-end tests (deselect with -m 'not integration')"]
addopts   = "-v --tb=short"

[tool.coverage.run]
source = ["tensorpress"]
omit   = ["tests/*", "examples/*"]
```

---

## `.github/workflows/ci.yml`

```yaml
name: CI

on:
  push:
    branches: [main, develop]
  pull_request:

jobs:
  test:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.10", "3.11", "3.12", "3.13"]

    steps:
      - uses: actions/checkout@v4

      - uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install (CPU-only torch for CI speed)
        run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          pip install -e ".[dev]"

      - name: Lint
        run: ruff check tensorpress/ examples/

      - name: Format check
        run: ruff format --check tensorpress/ examples/

      - name: Type check
        run: mypy tensorpress/ --ignore-missing-imports

      - name: Unit tests (fast)
        run: pytest tests/ -m "not integration" --cov=tensorpress --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v5
        with:
          token: ${{ secrets.CODECOV_TOKEN }}

  integration:
    runs-on: ubuntu-latest
    needs: test
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: |
          pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
          pip install -e ".[dev]"
          pytest tests/ -m integration -v

  publish:
    if: startsWith(github.ref, 'refs/tags/v')
    needs: test
    runs-on: ubuntu-latest
    environment: pypi
    permissions:
      id-token: write    # OIDC trusted publishing — no API token needed
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
      - run: pip install hatchling
      - run: python -m hatchling build
      - uses: pypa/gh-action-pypi-publish@release/v1
        # Uses OIDC — no PYPI_API_TOKEN secret required
```

> **Note on publish job:** Uses PyPI's trusted publisher (OIDC) — no API token needed.
> Set it up once at https://pypi.org/manage/account/publishing/.

---

## `MANIFEST.in`
No longer needed with hatchling — it uses `[tool.hatch.build]` in pyproject.toml instead:

```toml
# Add this to pyproject.toml
[tool.hatch.build.targets.sdist]
include = [
    "tensorpress/",
    "tests/",
    "examples/",
    "README.md",
    "LICENSE",
    "CHANGELOG.md",
]
```

---

## `CHANGELOG.md` (initial)

```markdown
# Changelog

All notable changes to this project will be documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] - Unreleased

### Added
- Tucker-2 decomposition for Conv2d layers (VBMF rank estimation)
- CPD (Canonical Polyadic) decomposition for Conv2d layers
- `Compressor` orchestrator with `CompressConfig`
- Layer selection: `"all"`, named list, regex, or callable predicate
- Rank modes: `"auto"` (VBMF), compression ratio float, or per-layer dict
- Optional BatchNorm insertion between factorized layers (`use_bn=True`)
- Optional fine-tuning after compression (`finetune=True`)
- `FineTuner` with Lightning Fabric backend (falls back to pure PyTorch)
- `CompressionResult` with `.report()`, `.compare()`, `.export()`
- PyTorch backend (`backends/pytorch.py`)
- CIFAR-10 + ResNet18 example notebook
- Quickstart notebook with tiny CNN
```

---

## Output Contract
- `pip install -e ".[dev]"` succeeds ✓
- `ruff check` and `ruff format --check` pass ✓
- `mypy tensorpress/` passes ✓
- `pytest tests/ -m "not integration"` green on Python 3.10–3.13 ✓
- `python -m hatchling build` produces valid wheel + sdist ✓
- GitHub Actions workflow with OIDC publish ✓
