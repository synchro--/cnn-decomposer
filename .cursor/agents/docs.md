# AGENT_DOCS — Documentation

## Goal
Produce a complete, user-facing documentation layer:
docstrings, mkdocs site, and a quickstart notebook.

---

## Docstring Standard

All public symbols must use **NumPy-style** docstrings.

```python
def compress(self, model, dataloader=None) -> CompressionResult:
    """
    Run the full compression pipeline on a model.

    Parameters
    ----------
    model : nn.Module
        The model to compress. Modified in-place.
    dataloader : DataLoader, optional
        Required if ``config.finetune=True``.

    Returns
    -------
    CompressionResult
        Compressed model + before/after metrics.

    Raises
    ------
    ValueError
        If ``config.finetune=True`` but no dataloader provided.

    Examples
    --------
    >>> from tensorpress import Compressor, CompressConfig
    >>> cfg = CompressConfig(method="tucker")
    >>> result = Compressor(cfg).compress(model, dataloader)
    >>> result.report()
    """
```

---

## `docs/mkdocs.yml`

```yaml
site_name: TensorPress
theme:
  name: material
  palette:
    scheme: slate
    primary: cyan

nav:
  - Home: index.md
  - Getting Started: getting_started.md
  - User Guide:
    - Configuration: guide/config.md
    - Layer Selection: guide/layers.md
    - Rank Estimation: guide/ranks.md
    - Fine-Tuning: guide/finetune.md
  - API Reference:
    - Compressor: api/compressor.md
    - CompressConfig: api/config.md
    - CompressionResult: api/result.md
    - Decompositions: api/decompositions.md
  - Roadmap: roadmap.md

plugins:
  - mkdocstrings:
      handlers:
        python:
          options:
            docstring_style: numpy
```

---

## `docs/getting_started.md`

Write this doc covering:
1. Installation (`pip install tensorpress`)
2. The 5-step user flow with runnable code
3. CompressConfig options table
4. Quick comparison: Tucker vs CPD (when to use which)
5. Saving and loading compressed models

---

## `docs/roadmap.md`

```markdown
# Roadmap

## v0.1 (current)
- Tucker & CPD for Conv2d
- PyTorch backend
- Auto rank estimation (VBMF)
- Optional fine-tuning

## v0.2
- SVD decomposition
- Rank sensitivity analysis
- `@compress_model` decorator

## v0.3
- TensorFlow backend
- JAX backend

## v1.0
- Structured pruning integration
- Quantization-aware compression
- ONNX export support
```

---

## `examples/quickstart.ipynb`

Create a Jupyter notebook with these cells:
1. Installation cell
2. Import + model setup (use torchvision ResNet18 if available, else tiny custom model)
3. `CompressConfig` examples (all three ranks modes)
4. Run compression, print `result.report()`
5. Side-by-side inference comparison
6. Export and reload

---

## `README.md`

Must include:
- Badges: PyPI version, CI status, coverage, Python versions
- 1-paragraph description
- Installation
- Minimal code example (the 5-step flow)
- Link to docs
- Contributing section
- License

---

## Output Contract
- Every public class and function has a NumPy docstring ✓
- `mkdocs build` succeeds ✓
- `examples/quickstart.ipynb` runs top-to-bottom without error ✓
- `README.md` has working code example ✓
