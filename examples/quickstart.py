"""TensorPress quickstart.

Compress a tiny vision CNN with Tucker or CPD decomposition, optionally
fine-tune it, and compare accuracy before and after compression.

Usage
-----
python examples/quickstart.py
python examples/quickstart.py --method cpd --epochs 3 --ft-epochs 1
"""

from __future__ import annotations

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from tensorpress import CompressConfig, Compressor
from tensorpress.config import FinetuneConfig

from examples.datasets import (
    format_supported_datasets,
    get_vision_dataloaders,
    supported_dataset_names,
)
from examples.output import make_results_path, save_run_results


class TinyCNN(nn.Module):
    """Small image classifier that trains quickly on CPU."""

    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def resolve_device() -> str:
    """Choose the best available PyTorch device."""
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def get_dataloaders(
    dataset: str = "cifar10",
    data_dir: str = "./data",
    batch_size: int = 128,
    train_size: int | None = 2048,
    val_size: int = 512,
    test_size: int | None = 1024,
) -> tuple[dict[str, DataLoader], int, int]:
    """Create small dataloaders for the quickstart workflow."""
    loaders, spec = get_vision_dataloaders(
        dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    return loaders, spec.channels, spec.num_classes


def train(model: nn.Module, loaders: dict[str, DataLoader], epochs: int, device: str) -> list[float]:
    """Train the baseline model with a plain PyTorch loop."""
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(epochs, 1))
    history: list[float] = []

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())

        scheduler.step()
        epoch_loss = running_loss / max(len(loaders["train"]), 1)
        history.append(epoch_loss)
        print(f"  epoch {epoch:2d}/{epochs}: loss={epoch_loss:.4f}")

    return history


def evaluate(model: nn.Module, loader: DataLoader, device: str) -> float:
    """Return classification accuracy as a percentage."""
    model = model.to(device)
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs, labels = inputs.to(device), labels.to(device)
            preds = torch.argmax(model(inputs), dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return 100.0 * correct / max(total, 1)


def parse_ranks(value: str) -> str | float:
    """Parse CLI rank value into a TensorPress rank spec."""
    if value == "auto":
        return value
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorPress quickstart")
    parser.add_argument("--dataset", default="cifar10", choices=supported_dataset_names())
    parser.add_argument("--list-datasets", action="store_true", help="list supported datasets and exit")
    parser.add_argument("--method", default="tucker", choices=["tucker", "cpd"])
    parser.add_argument("--ranks", default="auto", help="'auto' or a float ratio, for example 0.5")
    parser.add_argument("--epochs", type=int, default=1, help="baseline training epochs")
    parser.add_argument("--ft-epochs", type=int, default=1, help="post-compression fine-tune epochs")
    parser.add_argument("--use-bn", action="store_true", help="add BatchNorm after factors")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=2048)
    parser.add_argument("--val-size", type=int, default=512)
    parser.add_argument("--test-size", type=int, default=1024)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--export-path", default="compressed_tinycnn.pt")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="directory for timestamped JSON result dumps",
    )
    args = parser.parse_args()

    if args.list_datasets:
        print(format_supported_datasets())
        return

    device = resolve_device()
    print(
        f"\nTensorPress quickstart: device={device} dataset={args.dataset} "
        f"method={args.method} ranks={args.ranks}\n"
    )

    print(f"Loading {args.dataset}...")
    loaders, input_channels, num_classes = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )

    print(f"\nTraining TinyCNN for {args.epochs} epoch(s)...")
    model = TinyCNN(input_channels=input_channels, num_classes=num_classes)
    start = time.time()
    train(model, loaders, epochs=args.epochs, device=device)
    print(f"  training time: {time.time() - start:.1f}s")

    acc_before = evaluate(model, loaders["test"], device)
    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBaseline:   acc={acc_before:.2f}% params={params_before:,}")

    cfg = CompressConfig(
        method=args.method,
        layers="all",
        ranks=parse_ranks(args.ranks),
        use_bn=args.use_bn,
        finetune=args.ft_epochs > 0,
        finetune_config=FinetuneConfig(
            epochs=max(args.ft_epochs, 1),
            lr=1e-4,
            scheduler="cosine",
            use_amp=(device == "cuda"),
        ),
    )

    print(f"\nCompressing with {args.method.upper()}...")
    start = time.time()
    result = Compressor(cfg).compress(
        model,
        dataloader={"train": loaders["train"], "val": loaders["val"]},
    )
    print(f"  compression time: {time.time() - start:.1f}s")

    acc_after = evaluate(result, loaders["test"], device)
    ratio = result.trainable_params_before / max(result.trainable_params_after, 1)

    print("\nCompression report")
    result.report()
    print(f"\nBaseline:   acc={acc_before:.2f}% params={result.trainable_params_before:,}")
    print(f"Compressed: acc={acc_after:.2f}% params={result.trainable_params_after:,}")
    print(f"Compression ratio: {ratio:.2f}x")
    print(f"Accuracy delta: {acc_after - acc_before:+.2f} percentage points")

    result.export(args.export_path)
    results_path = save_run_results(
        make_results_path("quickstart", output_dir=args.output_dir),
        result=result,
        run={
            "example": "quickstart",
            "device": device,
            "dataset": args.dataset,
            "method": args.method,
            "ranks": args.ranks,
            "epochs": args.epochs,
            "ft_epochs": args.ft_epochs,
            "use_bn": args.use_bn,
            "batch_size": args.batch_size,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
        },
        metrics={
            "acc_before_pct": round(acc_before, 2),
            "acc_after_pct": round(acc_after, 2),
            "acc_delta_pct": round(acc_after - acc_before, 2),
            "compression_ratio_x": round(ratio, 2),
        },
        export_path=args.export_path,
    )
    print(f"\nSaved compressed model to {args.export_path}")
    print(f"Saved run results to {results_path}")


if __name__ == "__main__":
    main()
