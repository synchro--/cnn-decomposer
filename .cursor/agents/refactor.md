# AGENT_REFACTOR — Migrate Code Into Package Structure

## Pre-condition
AGENT_PYTORCH_FIXES must have completed successfully.
All bugs in source files are fixed before this agent runs.

## Goal
Reorganize the fixed source code into the `tensorpress/` package layout.
No new logic. No new features. Pure structural reorganization.

---

## Source → Target Mapping

### `mod_decomposer.py` is the PRIMARY source (cleaner, public/private split)
### `decompositions.py` is the SECONDARY source (extract BN-variant logic only)

---

## Step 1: Create `tensorpress/decompositions/base.py`

```python
from abc import ABC, abstractmethod
from typing import Any
import torch.nn as nn

class BaseDecomposition(ABC):
    """Abstract base class for all tensor decomposition strategies."""

    @abstractmethod
    def decompose(self, layer: nn.Conv2d, ranks: Any, use_bn: bool = False) -> nn.Sequential:
        """
        Decompose a Conv2d into a factorized nn.Sequential replacement.

        Parameters
        ----------
        layer : nn.Conv2d
            Original layer to compress.
        ranks : int | list[int]
            Rank(s) for the decomposition.
        use_bn : bool
            If True, insert BatchNorm2d between factorized layers.

        Returns
        -------
        nn.Sequential
            Drop-in replacement for the original layer.
        """
        ...

    @abstractmethod
    def estimate_ranks(self, layer: nn.Conv2d) -> list:
        """
        Auto-estimate ranks using VBMF.

        Returns
        -------
        list
            Estimated ranks [R1, R2] for Tucker or [R] for CPD.
        """
        ...
```

---

## Step 2: Create `tensorpress/decompositions/tucker.py`

**Source:** `mod_decomposer._tucker_layer_decomposition()` and
`mod_decomposer.estimate_tucker_ranks()`.
Also carry over the `_choose_compression` logic for Tucker (flag='Tucker2').

Key implementation notes:
- `decompose()` calls `_build_layers()` then optionally inserts BN
- BN variant: after `first_layer`, after `core_layer`, after `last_layer`
- The bias lives on `last_layer` only (copy from original layer if present)
- Weight assignment uses `.cpu().numpy()` everywhere (already fixed by AGENT_PYTORCH_FIXES)

```python
import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import partial_tucker
from .base import BaseDecomposition

class TuckerDecomposition(BaseDecomposition):

    def decompose(self, layer: nn.Conv2d, ranks: list, use_bn: bool = False) -> nn.Sequential:
        core, [last, first] = partial_tucker(
            layer.weight.data.cpu().numpy(),
            modes=[0, 1], ranks=ranks, init='svd'
        )

        first_layer = nn.Conv2d(
            in_channels=first.shape[0], out_channels=first.shape[1],
            kernel_size=1, stride=layer.stride, padding=0,
            dilation=layer.dilation, bias=False
        )
        core_layer = nn.Conv2d(
            in_channels=core.shape[1], out_channels=core.shape[0],
            kernel_size=layer.kernel_size, stride=layer.stride,
            padding=layer.padding, dilation=layer.dilation, bias=False
        )
        last_layer = nn.Conv2d(
            in_channels=last.shape[1], out_channels=last.shape[0],
            kernel_size=1, stride=layer.stride, padding=0,
            dilation=layer.dilation,
            bias=(layer.bias is not None)
        )
        if layer.bias is not None:
            last_layer.bias.data = layer.bias.data

        # assign weights — transpose to match PyTorch [out, in, h, w] format
        first_t = first.transpose((1, 0))
        first_layer.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(first_t.copy(), -1), -1)))
        last_layer.weight.data = torch.from_numpy(
            np.float32(np.expand_dims(np.expand_dims(last.copy(), -1), -1)))
        core_layer.weight.data = torch.from_numpy(np.float32(core.copy()))

        layers = [first_layer, core_layer, last_layer]

        if use_bn:
            layers = [
                first_layer, nn.BatchNorm2d(first.shape[1]),
                core_layer,  nn.BatchNorm2d(core.shape[0]),
                last_layer,  nn.BatchNorm2d(last.shape[0]),
            ]

        return nn.Sequential(*layers)

    def estimate_ranks(self, layer: nn.Conv2d) -> list:
        """VBMF rank estimation for Tucker-2 decomposition."""
        from VBMF import VBMF
        weights = layer.weight.data.cpu().numpy()
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        return [diag_0.shape[0], diag_1.shape[1]]
```

---

## Step 3: Create `tensorpress/decompositions/cpd.py`

**Source:** `mod_decomposer._cp_decomposition()` and
`mod_decomposer.estimate_cp_ranks()`.
Also carry over `_choose_compression` logic for CPD (flag='cpd').

Key implementation notes:
- CPD produces 4 layers: `[pointwise_s_to_r, depthwise_vertical, depthwise_horizontal, pointwise_r_to_t]`
- BN variant: insert `BatchNorm2d` after each of the 4 layers
- `parafac` new API: `_, factors = parafac(X, rank=rank, ...); last, first, vertical, horizontal = factors`
- For large tensors (`max(X.shape) >= 256`), use `init='random'`; else `init='svd'`
- Bias lives on `pointwise_r_to_t` only

```python
import numpy as np
import torch
import torch.nn as nn
import tensorly as tl
from tensorly.decomposition import parafac
from .base import BaseDecomposition

class CPDecomposition(BaseDecomposition):

    def decompose(self, layer: nn.Conv2d, ranks: int, use_bn: bool = False) -> nn.Sequential:
        X = layer.weight.data.cpu().numpy()
        rank = ranks if isinstance(ranks, int) else ranks[0]

        init = 'random' if max(X.shape) >= 256 else 'svd'
        _, factors = parafac(X, rank=rank, init=init)
        last, first, vertical, horizontal = factors

        first_pointwise = nn.Conv2d(
            first.shape[0], first.shape[1], kernel_size=1,
            stride=layer.stride, padding=0, dilation=layer.dilation, bias=False
        )
        depthwise_vertical = nn.Conv2d(
            vertical.shape[1], vertical.shape[1],
            kernel_size=(vertical.shape[0], 1),
            stride=layer.stride, padding=(layer.padding[0], 0),
            dilation=layer.dilation, groups=vertical.shape[1], bias=False
        )
        depthwise_horizontal = nn.Conv2d(
            horizontal.shape[1], horizontal.shape[1],
            kernel_size=(1, horizontal.shape[0]),
            stride=layer.stride, padding=(0, layer.padding[0]),
            dilation=layer.dilation, groups=horizontal.shape[1], bias=False
        )
        last_pointwise = nn.Conv2d(
            last.shape[1], last.shape[0], kernel_size=1,
            stride=layer.stride, padding=0, dilation=layer.dilation,
            bias=(layer.bias is not None)
        )
        if layer.bias is not None:
            last_pointwise.bias.data = layer.bias.data

        # Assign weights — transpose to PyTorch [out, in, h, w]
        depthwise_vertical.weight.data = torch.from_numpy(np.float32(
            np.expand_dims(np.expand_dims(vertical.transpose(1, 0), 1), -1)))
        depthwise_horizontal.weight.data = torch.from_numpy(np.float32(
            np.expand_dims(np.expand_dims(horizontal.transpose(1, 0), 1), 1)))
        first_pointwise.weight.data = torch.from_numpy(np.float32(
            np.expand_dims(np.expand_dims(first.transpose(1, 0), -1), -1)))
        last_pointwise.weight.data = torch.from_numpy(np.float32(
            np.expand_dims(np.expand_dims(last, -1), -1)))

        layers = [first_pointwise, depthwise_vertical, depthwise_horizontal, last_pointwise]

        if use_bn:
            layers = [
                first_pointwise,   nn.BatchNorm2d(first_pointwise.out_channels),
                depthwise_vertical, nn.BatchNorm2d(depthwise_vertical.out_channels),
                depthwise_horizontal, nn.BatchNorm2d(depthwise_horizontal.out_channels),
                last_pointwise,    nn.BatchNorm2d(last_pointwise.out_channels),
            ]

        return nn.Sequential(*layers)

    def estimate_ranks(self, layer: nn.Conv2d) -> list:
        """VBMF rank estimation for CP decomposition. Returns [rank]."""
        from VBMF import VBMF
        weights = layer.weight.data.cpu().numpy()
        unfold_0 = tl.base.unfold(weights, 0)
        unfold_1 = tl.base.unfold(weights, 1)
        _, diag_0, _, _ = VBMF.EVBMF(unfold_0)
        _, diag_1, _, _ = VBMF.EVBMF(unfold_1)
        rank = max(diag_0.shape[0], diag_1.shape[1])
        return [max(rank, 1)]  # guard against rank=0
```

---

## Step 4: Create `tensorpress/decompositions/__init__.py`

```python
from .base import BaseDecomposition
from .tucker import TuckerDecomposition
from .cpd import CPDecomposition

__all__ = ["BaseDecomposition", "TuckerDecomposition", "CPDecomposition"]
```

---

## Step 5: Move `custom_models.py` → `examples/custom_models.py`

This file contains test/research models, not library code.
Create `examples/` directory and move it there as-is.
Fix the one known bug: `torch.nn.init.xavier_uniform` → `torch.nn.init.xavier_uniform_`

---

## Step 6: What NOT to migrate

Do not carry these into `tensorpress/decompositions/`:
- `SVD_weights`, `FC_SVD_compression`, `conv1x1_SVD_compression` → future v0.2
- `tucker_xavier`, `cp_xavier_conv_layer` → superseded by `use_bn` config option
- All `print()` debug statements → use `logging.getLogger(__name__).debug()`
- All `logger.log_compression()` calls → handled by `CompressionResult`
- The `matlab=True` / `offline=True` code paths → out of scope for v1

---

## Output Contract

After this agent:
```python
from tensorpress.decompositions import TuckerDecomposition, CPDecomposition
import torch.nn as nn

layer = nn.Conv2d(16, 32, 3, padding=1, bias=True)
tucker = TuckerDecomposition()
ranks = tucker.estimate_ranks(layer)
result = tucker.decompose(layer, ranks)
assert isinstance(result, nn.Sequential)
assert result(torch.randn(1, 16, 8, 8)).shape == (1, 32, 8, 8)

cpd = CPDecomposition()
rank = cpd.estimate_ranks(layer)
result = cpd.decompose(layer, rank)
assert result(torch.randn(1, 16, 8, 8)).shape == (1, 32, 8, 8)
```
All assertions must pass. Output shapes must match input layer.
