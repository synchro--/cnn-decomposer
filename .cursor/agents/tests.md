# AGENT_TESTS — Test Suite

## Goal
Generate a comprehensive pytest suite covering unit and integration tests.
Tests must run without a GPU (use small models and CPU).

---

## `tests/conftest.py`

```python
import pytest
import torch
import torch.nn as nn

@pytest.fixture
def tiny_conv():
    """A minimal Conv2d for unit tests."""
    layer = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
    torch.nn.init.normal_(layer.weight)
    return layer

@pytest.fixture
def tiny_model():
    """A small sequential model with multiple Conv2d layers."""
    return nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.Conv2d(16, 32, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
    )

@pytest.fixture
def dummy_dataloader():
    """Yields a single batch of (input, target) for finetune tests."""
    inputs = torch.randn(4, 3, 32, 32)
    targets = torch.randint(0, 10, (4,))
    return [(inputs, targets)]
```

---

## `tests/test_tucker.py`

```python
def test_tucker_decompose_output_shape(tiny_conv):
    from tensorpress.decompositions import TuckerDecomposition
    decomp = TuckerDecomposition()
    ranks = decomp.estimate_ranks(tiny_conv)
    factorized = decomp.decompose(tiny_conv, ranks)
    
    x = torch.randn(1, 16, 8, 8)
    original_out = tiny_conv(x)
    factorized_out = factorized(x)
    assert original_out.shape == factorized_out.shape

def test_tucker_fewer_params(tiny_conv):
    ...  # factorized should have fewer params

def test_tucker_estimate_ranks_returns_tuple(tiny_conv):
    ...
```

---

## `tests/test_cpd.py`
Same structure as `test_tucker.py` but for `CPDecomposition`.

---

## `tests/test_config.py`

```python
def test_compress_config_defaults():
    from tensorpress import CompressConfig
    cfg = CompressConfig()
    assert cfg.method == "tucker"
    assert cfg.layers == "all"
    assert cfg.ranks == "auto"

def test_compress_config_json_roundtrip():
    cfg = CompressConfig(method="cpd", finetune=True)
    d = cfg.to_dict()
    restored = CompressConfig.from_dict(d)
    assert restored.method == cfg.method

def test_invalid_method_raises():
    with pytest.raises(ValueError):
        CompressConfig(method="bad").validate()

def test_layer_selector_all(tiny_model):
    from tensorpress.config import LayerSelector
    sel = LayerSelector("all")
    layers = sel.select(tiny_model)
    assert len(layers) == 2  # two Conv2d layers

def test_layer_selector_by_name(tiny_model):
    ...

def test_layer_selector_by_callable(tiny_model):
    ...
```

---

## `tests/test_compressor.py`

```python
def test_compressor_reduce_params(tiny_model):
    from tensorpress import Compressor, CompressConfig
    original_params = sum(p.numel() for p in tiny_model.parameters())
    cfg = CompressConfig(method="tucker", layers="all", ranks=0.5)
    result = Compressor(cfg).compress(tiny_model)
    assert result.compressed_params < original_params

def test_compressor_result_has_model(tiny_model):
    ...

def test_compressor_report_runs(tiny_model, capsys):
    ...

def test_compressor_export(tiny_model, tmp_path):
    path = str(tmp_path / "model.pt")
    result = Compressor(CompressConfig()).compress(tiny_model)
    result.export(path)
    assert os.path.exists(path)
```

---

## `tests/test_integration.py`

Full end-to-end pipeline test — slowest, tagged `@pytest.mark.integration`.

```python
@pytest.mark.integration
def test_full_pipeline_tucker(tiny_model, dummy_dataloader):
    cfg = CompressConfig(
        method="tucker",
        layers="all",
        ranks="auto",
        finetune=True,
        finetune_config=FinetuneConfig(epochs=1, lr=1e-3),
    )
    result = Compressor(cfg).compress(tiny_model, dataloader=dummy_dataloader)
    assert result.compression_ratio > 1.0
    result.report()   # should not raise
```

---

## CI Notes
- Run with `pytest tests/ -m "not integration"` for fast CI
- Run `pytest tests/ -m integration` separately or nightly
- Target: 80%+ coverage on `core/` and `decompositions/`

## Output Contract
- All unit tests pass on CPU ✓
- Integration test completes in < 60s on CPU ✓
- `pytest --co` shows all tests collected ✓
