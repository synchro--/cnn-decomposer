# AGENT_EXAMPLES — Training Loop Examples

## Goal
Create two runnable examples that demonstrate the full
compress → finetune → evaluate workflow:

1. **`examples/quickstart.py`** — tiny custom CNN on CIFAR-10, no torchvision model dep,
   runs in under 2 minutes on CPU. Entry point for new users.

2. **`examples/cifar10_resnet18.py`** — realistic ResNet18 + CIFAR-10 workflow,
   shows the library's real compression gains, GPU-aware.

3. **`examples/quickstart.ipynb`** — notebook wrapping the quickstart script
   with markdown explanations and inline result display.

Both scripts use pure stdlib + tensorpress only (no Lightning required in the example
itself — the library's FineTuner handles that internally).

---

## `examples/quickstart.py`

Complete, runnable script. No GPU required. Should complete in ~90s on a modern CPU.

```python
"""
TensorPress Quickstart
======================
Compress a tiny custom CNN trained on CIFAR-10 using Tucker decomposition,
then compare accuracy before and after.

Usage:
    python examples/quickstart.py
    python examples/quickstart.py --method cpd --epochs 3
"""

import argparse
import time
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from tensorpress import Compressor, CompressConfig
from tensorpress.config import FinetuneConfig


# ── 1. Tiny model ──────────────────────────────────────────────────────────────

class TinyCNN(nn.Module):
    """Small CIFAR-10 classifier — fast to train, easy to compress."""

    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),   # ← will be compressed
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1),  # ← will be compressed
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, 3, padding=1),  # ← will be compressed
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 10),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


# ── 2. Data ────────────────────────────────────────────────────────────────────

def get_dataloaders(data_dir: str = "./data", batch_size: int = 128):
    transform_train = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    transform_val = T.Compose([
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])

    full_train = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform_train
    )
    val_set = torchvision.datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform_val
    )
    test_set = torchvision.datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform_val
    )

    # 45k train / 5k val split
    n_val = 5_000
    train_set, _ = random_split(
        full_train, [len(full_train) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )
    _, val_subset = random_split(
        val_set, [len(val_set) - n_val, n_val],
        generator=torch.Generator().manual_seed(42)
    )

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=2),
        "val":   DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=2),
        "test":  DataLoader(test_set,  batch_size=batch_size, shuffle=False, num_workers=2),
    }


# ── 3. Training loop ───────────────────────────────────────────────────────────

def train(model, loaders, epochs: int = 10, device: str = "cpu") -> list[float]:
    """
    Standard PyTorch training loop — completely independent of TensorPress.
    TensorPress only touches the model at compression time.
    """
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        scheduler.step()
        epoch_loss = running_loss / len(loaders["train"])
        history.append(epoch_loss)
        print(f"  Epoch {epoch:2d}/{epochs} — loss: {epoch_loss:.4f}")

    return history


def evaluate(model, loader, device: str = "cpu") -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


# ── 4. Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TensorPress quickstart")
    parser.add_argument("--method",  default="tucker", choices=["tucker", "cpd"])
    parser.add_argument("--ranks",   default="auto",
                        help="'auto', a float ratio (e.g. 0.5), or skip for VBMF")
    parser.add_argument("--epochs",  type=int, default=10,  help="pre-train epochs")
    parser.add_argument("--ft-epochs", type=int, default=3, help="fine-tune epochs")
    parser.add_argument("--use-bn",  action="store_true",   help="add BN after factorized layers")
    parser.add_argument("--data-dir", default="./data")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\n{'='*60}")
    print(f"  TensorPress Quickstart — device: {device}")
    print(f"  method={args.method}  ranks={args.ranks}  use_bn={args.use_bn}")
    print(f"{'='*60}\n")

    # Step 1 — data
    print("► Loading CIFAR-10...")
    loaders = get_dataloaders(args.data_dir)

    # Step 2 — build and train the base model
    print(f"\n► Pre-training TinyCNN for {args.epochs} epochs...")
    model = TinyCNN()
    t0 = time.time()
    train(model, loaders, epochs=args.epochs, device=device)
    print(f"  Done in {time.time()-t0:.1f}s")

    acc_before = evaluate(model, loaders["test"], device=device)
    params_before = sum(p.numel() for p in model.parameters())
    print(f"\n  ✓ Baseline  — test acc: {acc_before:.2f}%  params: {params_before:,}")

    # Step 3 — compress with TensorPress  ← THE INTERESTING PART
    print(f"\n► Compressing with TensorPress ({args.method.upper()})...")

    ranks = args.ranks
    if ranks not in ("auto",):
        try:
            ranks = float(ranks)
        except ValueError:
            pass

    cfg = CompressConfig(
        method=args.method,
        layers="all",
        ranks=ranks,
        use_bn=args.use_bn,
        finetune=args.ft_epochs > 0,
        finetune_config=FinetuneConfig(
            epochs=args.ft_epochs,
            lr=1e-4,
            scheduler="cosine",
            use_amp=(device == "cuda"),
        ),
    )

    t1 = time.time()
    result = Compressor(cfg).compress(
        model,
        dataloader={"train": loaders["train"], "val": loaders["val"]},
    )
    print(f"  Done in {time.time()-t1:.1f}s")

    # Step 4 — compare
    acc_after = evaluate(result.model, loaders["test"], device=device)

    print("\n" + "="*60)
    result.report()
    print(f"\n  Baseline  — test acc: {acc_before:.2f}%  params: {params_before:,}")
    print(f"  Compressed— test acc: {acc_after:.2f}%  params: {result.compressed_params:,}")
    print(f"  Δ accuracy: {acc_after - acc_before:+.2f}pp")
    print("="*60)

    result.export("compressed_tinycnn.pt")
    print("\n  Model saved to compressed_tinycnn.pt")


if __name__ == "__main__":
    main()
```

---

## `examples/cifar10_resnet18.py`

Realistic example — pretrained ResNet18 fine-tuned on CIFAR-10, then compressed.
Shows the library working on a production-grade model.

```python
"""
TensorPress — CIFAR-10 + ResNet18
==================================
1. Load a torchvision ResNet18 pretrained on ImageNet
2. Fine-tune the classifier head on CIFAR-10 (fast — only 5 epochs)
3. Compress ALL convolutional layers with Tucker decomposition
4. Fine-tune the compressed model for 3 epochs to recover accuracy
5. Print a full before/after report

Usage:
    python examples/cifar10_resnet18.py
    python examples/cifar10_resnet18.py --no-pretrained --epochs 20
"""

import argparse
import torch
import torch.nn as nn
import torchvision
import torchvision.models as tvm
import torchvision.transforms as T
from torch.utils.data import DataLoader, random_split

from tensorpress import Compressor, CompressConfig
from tensorpress.config import FinetuneConfig


def get_dataloaders(data_dir="./data", batch_size=128, img_size=224):
    # ResNet expects 224x224 for ImageNet weights; use 32 for speed if training from scratch
    resize = T.Resize(img_size) if img_size != 32 else nn.Identity()

    transform = T.Compose([
        resize,
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    val_transform = T.Compose([
        resize,
        T.ToTensor(),
        T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_ds = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=transform)
    val_ds   = torchvision.datasets.CIFAR10(data_dir, train=True,  download=True, transform=val_transform)
    test_ds  = torchvision.datasets.CIFAR10(data_dir, train=False, download=True, transform=val_transform)

    n_val = 5_000
    g = torch.Generator().manual_seed(42)
    train_set, _ = random_split(train_ds, [len(train_ds) - n_val, n_val], generator=g)
    _, val_set   = random_split(val_ds,   [len(val_ds)   - n_val, n_val], generator=g)

    return {
        "train": DataLoader(train_set, batch_size=batch_size, shuffle=True,  num_workers=4, pin_memory=True),
        "val":   DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
        "test":  DataLoader(test_ds,   batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True),
    }


def build_model(pretrained: bool = True, num_classes: int = 10):
    weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, device) -> float:
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            _, preds = torch.max(model(inputs), 1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100.0 * correct / total


def head_finetune(model, loaders, epochs, device):
    """Quickly fine-tune only the final FC layer (feature extractor frozen)."""
    for p in model.parameters():
        p.requires_grad = False
    for p in model.fc.parameters():
        p.requires_grad = True

    opt = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    model.to(device)

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            opt.zero_grad()
            criterion(model(inputs), labels).backward()
            opt.step()
        acc = evaluate(model, loaders["val"], device)
        print(f"  Head ft epoch {epoch}/{epochs} — val acc: {acc:.2f}%")

    # Unfreeze all for compression + full fine-tuning later
    for p in model.parameters():
        p.requires_grad = True


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--head-epochs",  type=int, default=3)
    parser.add_argument("--ft-epochs",    type=int, default=3)
    parser.add_argument("--method",       default="tucker", choices=["tucker", "cpd"])
    parser.add_argument("--data-dir",     default="./data")
    args = parser.parse_args()

    device = (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )
    print(f"\nDevice: {device}  |  pretrained={args.pretrained}  |  method={args.method}\n")

    img_size = 224 if args.pretrained else 32
    loaders  = get_dataloaders(args.data_dir, img_size=img_size)
    model    = build_model(pretrained=args.pretrained)

    # ── Step 1: Head fine-tune (fast) ────────────────────────────────────
    print(f"► Head fine-tuning for {args.head_epochs} epochs...")
    head_finetune(model, loaders, epochs=args.head_epochs, device=device)

    acc_before = evaluate(model, loaders["test"], device)
    n_before   = sum(p.numel() for p in model.parameters())
    print(f"\n  ✓ Baseline — test acc: {acc_before:.2f}%  params: {n_before:,}\n")

    # ── Step 2: Compress ─────────────────────────────────────────────────
    print(f"► Compressing with TensorPress ({args.method.upper()})...")
    cfg = CompressConfig(
        method=args.method,
        # Skip the first conv (stem) and last layer — only compress the residual blocks
        layers=lambda name, mod: "layer" in name,
        ranks="auto",
        use_bn=False,
        finetune=True,
        finetune_config=FinetuneConfig(
            epochs=args.ft_epochs,
            lr=5e-5,
            scheduler="cosine",
            use_amp=(device == "cuda"),
        ),
    )

    result = Compressor(cfg).compress(
        model,
        dataloader={"train": loaders["train"], "val": loaders["val"]},
    )

    # ── Step 3: Report ───────────────────────────────────────────────────
    acc_after = evaluate(result.model, loaders["test"], device)

    print("\n" + "="*70)
    result.report()
    print(f"\n  Baseline   — test acc: {acc_before:.2f}%   params: {n_before:,}")
    print(f"  Compressed — test acc: {acc_after:.2f}%   params: {result.compressed_params:,}")
    print(f"  Compression ratio: {result.compression_ratio:.2f}x")
    print(f"  Accuracy delta:    {acc_after - acc_before:+.2f}pp")
    print("="*70)

    result.export("compressed_resnet18.pt")
    print("\n  Model saved to compressed_resnet18.pt")


if __name__ == "__main__":
    main()
```

---

## `examples/quickstart.ipynb` structure

Cursor should generate this notebook from `quickstart.py` with these sections:

1. **Installation** — `pip install tensorpress[torch,rich,flops]`
2. **The idea** — 3-sentence explanation of Tucker/CPD with a diagram cell
3. **Data & model** — inline cells from `quickstart.py`
4. **Train the baseline** — run `train()`, plot loss curve with matplotlib
5. **Compress** — the `CompressConfig` + `Compressor` cells, annotated
6. **Report** — `result.report()` output + side-by-side accuracy bar chart
7. **Save & load** — `result.export()` + reload demo

---

## Key design notes for AGENT to follow

- Both scripts are **self-contained** — they import from `tensorpress` only.
- The training loop in `quickstart.py` is **intentionally plain PyTorch** — this
  demonstrates that TensorPress is framework-agnostic from the user's perspective.
  The user writes their own training loop. TensorPress only handles compression.
- The `FineTuner` inside `CompressConfig(finetune=True)` uses Fabric internally,
  but the user never sees that — it's an implementation detail.
- The `layers=lambda name, mod: "layer" in name` in `cifar10_resnet18.py` demonstrates
  the **callable layer selector** which is a key differentiator vs. competing tools.

---

## Output Contract
- `python examples/quickstart.py` runs end-to-end on CPU without error ✓
- `python examples/quickstart.py --method cpd --epochs 3 --ft-epochs 1` runs fast (CI-safe) ✓
- `python examples/cifar10_resnet18.py --no-pretrained --head-epochs 1 --ft-epochs 1` runs on CPU ✓
- All imports resolve with `pip install tensorpress[torch]` ✓
- Notebook cells all execute top-to-bottom without error ✓
