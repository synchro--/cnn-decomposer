# TensorPress — Cursor AI Agents

This file drives the full transformation of the existing research code into a
clean, installable, framework-agnostic compression library.

**Read this file first. Then run agents strictly in order.**

---

## Source Files Inventory

| File | Role | Status |
|---|---|---|
| `mod_decomposer.py` | Main decomposer module (public/private split) | ✅ Primary source — migrate this |
| `decompositions.py` | Older version with more variants | ⚠️ Secondary — extract BN logic only |
| `pytorch_utils.py` | Training helpers, weight I/O, data loaders | 🔧 Split and fix |
| `training_algo.py` | Training loops, test functions | 🔧 Adapt into FineTuner |
| `custom_models.py` | Research test models | 📦 Move to `examples/` |

---

## Version Baseline (verified Feb 2026)

| Package | Min version | Notes |
|---|---|---|
| Python | 3.10 | PyTorch 2.10 dropped 3.9 |
| torch | 2.10.0 | |
| torchvision | 0.25.0 | |
| tensorly | 0.8.1 | `_, factors = parafac(...)` new API |
| numpy | 2.0 | |
| scipy | 1.14 | VBMF SVD |
| torchinfo | 1.8.0 | **optional** — FLOPs counting |
| rich | 13.9 | **optional** — pretty `result.report()` |
| ruff | 0.9 | linter + formatter |
| pytest | 8.3 | |
| mypy | 1.13 | |
| hatchling | 1.27 | build backend |

---

## Execution Order

```
AGENT_PYTORCH_FIXES → AGENT_REFACTOR → AGENT_CONFIG → AGENT_BACKEND
  → AGENT_CORE → AGENT_UTILS → AGENT_TESTS → AGENT_EXAMPLES
  → AGENT_PACKAGING → AGENT_DOCS
```

**AGENT_PYTORCH_FIXES runs first** — it must fix all broken/deprecated APIs
before any migration happens, so subsequent agents work on clean code.

---

## Agent Index

| Agent | File | Phase | Description |
|---|---|---|---|
| AGENT_PYTORCH_FIXES | `.cursor/agents/pytorch_fixes.md` | 0 – Fix | Fix all deprecated/broken PyTorch & tensorly APIs |
| AGENT_REFACTOR | `.cursor/agents/refactor.md` | 1 – Migrate | Move code into new package structure |
| AGENT_CONFIG | `.cursor/agents/config.md` | 2 – Schema | Build CompressConfig + LayerSelector |
| AGENT_BACKEND | `.cursor/agents/backend.md` | 3 – Framework | PyTorchBackend ABC |
| AGENT_CORE | `.cursor/agents/core.md` | 4 – Orchestrate | Compressor + FineTuner (pure PyTorch) |
| AGENT_UTILS | `.cursor/agents/utils.md` | 5 – Utils | Metrics, profiling, serialization |
| AGENT_TESTS | `.cursor/agents/tests.md` | 6 – Tests | pytest suite |
| AGENT_EXAMPLES | `.cursor/agents/examples.md` | 7 – Examples | TinyCNN quickstart + ResNet18/CIFAR-10 full example |
| AGENT_PACKAGING | `.cursor/agents/packaging.md` | 8 – Release | pyproject.toml + CI (OIDC publish) |
| AGENT_DOCS | `.cursor/agents/docs.md` | 9 – Docs | Docstrings + mkdocs |

---

## Global Rules (apply to all agents)

- **Never delete logic** — move and adapt only.
- All public symbols must have NumPy-style docstrings.
- Type hints everywhere (Python ≥ 3.9).
- No `torch` imports at the top level of `tensorpress/__init__.py`.
- All decomposition math lives in `tensorpress/decompositions/` — never in core or backends.
- Use `dataclasses` for config objects — JSON-serializable.
- Every new module needs a corresponding test file.
- CI must stay green after each agent's work.

---

## Target Public API (do NOT break this contract)

```python
from tensorpress import Compressor, CompressConfig

model = MyModel()

cfg = CompressConfig(
    method="tucker",        # "tucker" | "cpd"
    layers="all",           # "all" | list[str] | regex str | callable
    ranks="auto",           # "auto" | float (compression ratio) | dict[str, int]
    use_bn=False,           # insert BatchNorm between factorized layers
    finetune=False,
    finetune_config=dict(epochs=5, lr=1e-4),
)

compressor = Compressor(cfg)
result = compressor.compress(model, dataloader=train_loader)

result.report()             # rich before/after table
result.compare()            # returns metrics dict
result.export("compressed.pt")
```

---

## Target Directory Layout

```
tensorpress/
├── __init__.py                  ← Compressor, CompressConfig, CompressionResult
├── config/
│   ├── __init__.py
│   ├── schema.py                ← CompressConfig, FinetuneConfig dataclasses
│   ├── selectors.py             ← LayerSelector
│   └── rank_estimators.py       ← RankEstimator, VBMFEstimator
├── core/
│   ├── __init__.py
│   ├── compressor.py            ← Compressor (orchestrator)
│   ├── replacer.py              ← LayerReplacer
│   ├── finetuner.py             ← FineTuner (from training_algo.py)
│   ├── result.py                ← CompressionResult
│   └── pipeline.py              ← run_pipeline()
├── decompositions/
│   ├── __init__.py
│   ├── base.py                  ← BaseDecomposition (ABC)
│   ├── tucker.py                ← TuckerDecomposition
│   └── cpd.py                   ← CPDecomposition
├── backends/
│   ├── __init__.py
│   ├── base.py                  ← BaseBackend (ABC)
│   └── pytorch.py               ← PyTorchBackend
└── utils/
    ├── __init__.py
    ├── metrics.py
    ├── profiler.py
    └── serialization.py

tests/
├── conftest.py
├── test_tucker.py
├── test_cpd.py
├── test_config.py
├── test_compressor.py
└── test_integration.py

examples/
├── quickstart.ipynb
└── custom_models.py             ← moved here from root (not library code)
```
