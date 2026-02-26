# AGENT_CORE — Orchestration Engine

## Goal
Build `Compressor`, `LayerReplacer`, `FineTuner`, and `CompressionResult`.

**Key change vs. original plan:** `FineTuner` uses **Lightning Fabric** when available,
falling back to pure PyTorch. Fabric adds device dispatch, AMP, and multi-GPU
without forcing a `LightningModule` structure — the loop stays plain Python.

---

## Step 1: `tensorpress/core/compressor.py`

```python
from __future__ import annotations
from tensorpress.config import CompressConfig
from tensorpress.backends import get_backend
from tensorpress.decompositions import TuckerDecomposition, CPDecomposition
from .pipeline import run_pipeline

_DECOMP_MAP = {
    "tucker": TuckerDecomposition,
    "cpd": CPDecomposition,
}

class Compressor:
    """
    Main entry point for model compression.

    Parameters
    ----------
    config : CompressConfig
        Compression strategy and options.

    Examples
    --------
    >>> from tensorpress import Compressor, CompressConfig
    >>> cfg = CompressConfig(method="tucker", layers="all", ranks="auto")
    >>> result = Compressor(cfg).compress(model, dataloader=loader)
    >>> result.report()
    """

    def __init__(self, config: CompressConfig):
        config.validate()
        self.config = config
        self._backend = get_backend(config._framework)
        self._decomp = _DECOMP_MAP[config.method]()

    def compress(self, model, dataloader=None):
        """
        Run the full compression pipeline.

        Parameters
        ----------
        model : nn.Module
            Model to compress (modified in-place).
        dataloader : DataLoader | dict[str, DataLoader], optional
            Required when ``config.finetune=True``.
            Pass a dict ``{"train": ..., "val": ...}`` for validation tracking.

        Returns
        -------
        CompressionResult
        """
        if self.config.finetune and dataloader is None:
            raise ValueError("dataloader is required when finetune=True")
        return run_pipeline(model, self.config, self._backend, self._decomp, dataloader)
```

---

## Step 2: `tensorpress/core/replacer.py`

```python
import logging
from tensorpress.backends.base import BaseBackend
from tensorpress.decompositions.base import BaseDecomposition

logger = logging.getLogger(__name__)

class LayerReplacer:
    """Replaces Conv2d layers with their factorized equivalents in-place."""

    def __init__(self, backend: BaseBackend, decomp: BaseDecomposition, use_bn: bool = False):
        self.backend = backend
        self.decomp = decomp
        self.use_bn = use_bn

    def replace_layers(self, model, selected_layers: list, rank_map: dict) -> dict:
        """
        Parameters
        ----------
        selected_layers : list[tuple[str, nn.Conv2d]]
        rank_map : dict[str, int | list[int]]

        Returns
        -------
        dict
            Per-layer stats: original_params, compressed_params, ranks.
        """
        stats = {}
        for name, layer in selected_layers:
            ranks = rank_map[name]
            logger.debug("Decomposing layer '%s' with ranks %s", name, ranks)
            factorized = self.decomp.decompose(layer, ranks, use_bn=self.use_bn)
            self.backend.replace_module(model, name, factorized)
            orig   = sum(p.numel() for p in layer.parameters())
            compr  = sum(p.numel() for p in factorized.parameters())
            stats[name] = {
                "original_params": orig,
                "compressed_params": compr,
                "ranks": ranks,
                "ratio": orig / max(compr, 1),
            }
            logger.debug("  → %.2fx compression for '%s'", stats[name]["ratio"], name)
        return stats
```

---

## Step 3: `tensorpress/core/finetuner.py`

Pure PyTorch only. No optional dependencies.
Fabric support is planned as a future demo/extension — do not add it here.

```python
from __future__ import annotations
import copy
import logging

logger = logging.getLogger(__name__)


class FineTuner:
    """
    Post-compression fine-tuning loop (pure PyTorch).

    Parameters
    ----------
    config : FinetuneConfig
    backend : BaseBackend

    Examples
    --------
    >>> tuner = FineTuner(config, backend)
    >>> history = tuner.finetune(model, {"train": train_dl, "val": val_dl})
    """

    def __init__(self, config, backend):
        self.config = config
        self.backend = backend

    def finetune(self, model, dataloader, loss_fn=None) -> list[float]:
        """
        Run fine-tuning.

        Parameters
        ----------
        model : nn.Module
        dataloader : DataLoader | dict[str, DataLoader]
            Single loader (train only) or dict with ``"train"`` / ``"val"`` keys.
        loss_fn : callable, optional
            Defaults to ``nn.CrossEntropyLoss()``.

        Returns
        -------
        list[float]
            Training loss per epoch.
        """
        import torch
        import torch.nn as nn
        import torch.optim as optim

        device = torch.device(
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )
        logger.info("FineTuner: device=%s  epochs=%d  lr=%g", device, self.config.epochs, self.config.lr)
        model = model.to(device)

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        train_loader, val_loader = self._split_loaders(dataloader)
        scheduler = self._build_scheduler(optimizer)

        # AMP — only on CUDA, opt-in via config.use_amp
        use_amp = self.config.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        history = []

        for epoch in range(self.config.epochs):
            model.train()
            running_loss = 0.0

            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                optimizer.zero_grad()

                if use_amp:
                    with torch.amp.autocast("cuda"):
                        loss = loss_fn(model(inputs), labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = loss_fn(model(inputs), labels)
                    loss.backward()
                    optimizer.step()

                running_loss += loss.item()

            if scheduler:
                scheduler.step()

            epoch_loss = running_loss / len(train_loader)
            history.append(epoch_loss)
            logger.info("Epoch %d/%d — loss: %.4f", epoch + 1, self.config.epochs, epoch_loss)

            if val_loader:
                acc = self._validate(model, val_loader, device)
                logger.info("  val acc: %.4f", acc)
                if acc > best_acc:
                    best_acc = acc
                    best_wts = copy.deepcopy(model.state_dict())

        if val_loader:
            model.load_state_dict(best_wts)  # restore best checkpoint
        return history

    # ── Helpers ───────────────────────────────────────────────────────────

    @staticmethod
    def _split_loaders(dataloader):
        if isinstance(dataloader, dict):
            return dataloader["train"], dataloader.get("val")
        return dataloader, None

    def _build_scheduler(self, optimizer):
        import torch.optim.lr_scheduler as sched
        name = self.config.scheduler
        if name == "cosine":
            return sched.CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        if name == "step":
            return sched.StepLR(optimizer, step_size=max(1, self.config.epochs // 3), gamma=0.1)
        return None  # "none"

    @staticmethod
    def _validate(model, val_loader, device) -> float:
        import torch
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                _, preds = torch.max(model(inputs), 1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        model.train()
        return correct / max(total, 1)
```

---

## Step 4: Update `tensorpress/config/schema.py` — add new FinetuneConfig fields

The schema needs `use_amp`, `weight_decay`, and `scheduler` to match `FineTuner`:

```python
@dataclass
class FinetuneConfig:
    epochs: int = 5
    lr: float = 1e-4
    weight_decay: float = 1e-5
    scheduler: Literal["cosine", "step", "none"] = "cosine"
    use_amp: bool = False    # Automatic Mixed Precision (CUDA only; Fabric handles it too)
```

Update `CompressConfig` to also expose `use_bn`:

```python
@dataclass
class CompressConfig:
    method: Literal["tucker", "cpd"] = "tucker"
    layers: str | list[str] | Callable = "all"
    ranks: str | float | dict = "auto"
    use_bn: bool = False          # insert BatchNorm between factorized sub-layers
    finetune: bool = False
    finetune_config: FinetuneConfig = field(default_factory=FinetuneConfig)
    _framework: str = field(default="pytorch", init=False, repr=False)
```

---

## Step 5: `tensorpress/core/result.py` — unchanged from previous spec

See previous `core.md` version. No changes needed here.

---

## Step 6: `tensorpress/core/pipeline.py` — unchanged from previous spec

See previous `core.md` version. No changes needed here.

---

## Output Contract
- `Compressor(cfg).compress(model)` runs end-to-end on CPU ✓
- `FineTuner` uses Fabric when `lightning` is installed, pure PyTorch otherwise ✓
- Both paths accept `DataLoader` or `{"train": ..., "val": ...}` dict ✓
- AMP works on CUDA via both paths ✓
- MPS device detected automatically in pure PyTorch fallback ✓
- `CompressionResult.report()` works with and without `rich` installed ✓
- `pipeline.py` has no direct `torch` imports — all via `backend.*` ✓
