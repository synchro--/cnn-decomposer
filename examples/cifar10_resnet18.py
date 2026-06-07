"""TensorPress ResNet18 vision example.

Fine-tune a ResNet18 classifier on a small torchvision dataset, compress
residual block convolutions, fine-tune the compressed model, and report
before/after metrics.

Usage
-----
python examples/cifar10_resnet18.py
python examples/cifar10_resnet18.py --no-pretrained --head-epochs 1 --ft-epochs 1
"""

from __future__ import annotations

import argparse

import torch
import torch.nn as nn
import torchvision.models as tvm
from torch.utils.data import DataLoader

from tensorpress import CompressConfig, Compressor
from tensorpress.config import FinetuneConfig

from examples.datasets import (
    format_supported_datasets,
    get_vision_dataloaders,
    supported_dataset_names,
)
from examples.output import make_results_path, save_run_results


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
    img_size: int = 224,
    train_size: int | None = 4096,
    val_size: int = 1024,
    test_size: int | None = 2048,
) -> tuple[dict[str, DataLoader], int]:
    """Create dataloaders sized for an example ResNet18 run."""
    loaders, spec = get_vision_dataloaders(
        dataset,
        data_dir=data_dir,
        batch_size=batch_size,
        image_size=img_size,
        output_channels=3,
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
    )
    return loaders, spec.num_classes


def build_model(pretrained: bool = True, num_classes: int = 10) -> nn.Module:
    """Build ResNet18 and replace the classifier for CIFAR-10."""
    weights = tvm.ResNet18_Weights.DEFAULT if pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


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


def head_finetune(model: nn.Module, loaders: dict[str, DataLoader], epochs: int, device: str) -> None:
    """Fine-tune only the final fully connected layer."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, epochs + 1):
        model.train()
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        val_acc = evaluate(model, loaders["val"], device)
        print(f"  head epoch {epoch:2d}/{epochs}: val_acc={val_acc:.2f}%")

    for parameter in model.parameters():
        parameter.requires_grad = True


def main() -> None:
    parser = argparse.ArgumentParser(description="TensorPress ResNet18 vision example")
    parser.add_argument("--dataset", default="cifar10", choices=supported_dataset_names())
    parser.add_argument("--list-datasets", action="store_true", help="list supported datasets and exit")
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--head-epochs", type=int, default=1)
    parser.add_argument("--ft-epochs", type=int, default=1)
    parser.add_argument("--method", default="tucker", choices=["tucker", "cpd"])
    parser.add_argument(
        "--compression",
        type=float,
        default=0.1,
        help=(
            "Keep-fraction target in (0, 1] for the compressed conv layers. "
            "Lower keeps fewer params -> smaller rank -> far less decomposition "
            "RAM (important for CPD, which OOMs at large ranks on host memory)."
        ),
    )
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--export-path", default="compressed_resnet18.pt")
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
    img_size = 224 if args.pretrained else 32
    print(
        f"\nTensorPress ResNet18: device={device} dataset={args.dataset} "
        f"pretrained={args.pretrained}\n"
    )

    loaders, num_classes = get_dataloaders(
        dataset=args.dataset,
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        img_size=img_size,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
    )
    model = build_model(pretrained=args.pretrained, num_classes=num_classes)

    print(f"Head fine-tuning for {args.head_epochs} epoch(s)...")
    head_finetune(model, loaders, epochs=args.head_epochs, device=device)

    acc_before = evaluate(model, loaders["test"], device)
    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nBaseline: acc={acc_before:.2f}% params={params_before:,}\n")

    cfg = CompressConfig(
        method=args.method,
        layers=lambda name, module: "layer" in name and isinstance(module, nn.Conv2d),
        compression=args.compression,
        use_bn=False,
        finetune=args.ft_epochs > 0,
        finetune_config=FinetuneConfig(
            epochs=max(args.ft_epochs, 1),
            lr=5e-5,
            scheduler="cosine",
            use_amp=(device == "cuda"),
        ),
    )

    print(f"Compressing residual block convolutions with {args.method.upper()}...")
    result = Compressor(cfg).compress(
        model,
        dataloader={"train": loaders["train"], "val": loaders["val"]},
    )

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
        make_results_path("resnet18", output_dir=args.output_dir),
        result=result,
        run={
            "example": "cifar10_resnet18",
            "device": device,
            "dataset": args.dataset,
            "method": args.method,
            "compression": args.compression,
            "pretrained": args.pretrained,
            "head_epochs": args.head_epochs,
            "ft_epochs": args.ft_epochs,
            "batch_size": args.batch_size,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
            "image_size": img_size,
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
