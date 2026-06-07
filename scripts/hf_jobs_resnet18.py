# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "torch>=2.10",
#   "torchvision>=0.25",
#   "tensorly>=0.8.1",
#   "numpy>=2.0",
#   "scipy>=1.14",
#   "huggingface-hub",
# ]
# ///
"""Run TensorPress ResNet18 compression on Hugging Face Jobs.

Memory-safe defaults for cloud GPUs (batch_size=32, AMP on CUDA).
Results are pushed to a Hub model repo when HF_TOKEN is set.

Usage (local submitter)
-----------------------
python scripts/submit_resnet18_job.py
"""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
from pathlib import Path

INSTALL_SPEC = (
    "tensorpress[torch] @ git+https://github.com/synchro--/cnn-decomposer.git@feature"
)


def install_tensorpress() -> None:
    """Install TensorPress from the public feature branch."""
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", INSTALL_SPEC],
    )


def resolve_device() -> str:
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def run_experiment(args: argparse.Namespace) -> dict:
    import torch
    import torch.nn as nn
    import torchvision.models as tvm
    from torch.utils.data import DataLoader

    from examples.datasets import get_vision_dataloaders
    from tensorpress import CompressConfig, Compressor
    from tensorpress.config import FinetuneConfig

    device = resolve_device()
    print(f"device={device} dataset={args.dataset} method={args.method} ranks={args.ranks}")

    loaders, spec = get_vision_dataloaders(
        args.dataset,
        data_dir=str(Path(args.data_dir)),
        batch_size=args.batch_size,
        image_size=224 if args.pretrained else 32,
        output_channels=3,
        train_size=args.train_size,
        val_size=args.val_size,
        test_size=args.test_size,
        num_workers=args.num_workers,
    )

    weights = tvm.ResNet18_Weights.DEFAULT if args.pretrained else None
    model = tvm.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, spec.num_classes)

    # Head fine-tune (classifier only)
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc.parameters():
        parameter.requires_grad = True

    model = model.to(device)
    optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.head_epochs + 1):
        model.train()
        for inputs, labels in loaders["train"]:
            inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(inputs), labels)
            loss.backward()
            optimizer.step()
        print(f"  head epoch {epoch}/{args.head_epochs} complete")

    for parameter in model.parameters():
        parameter.requires_grad = True

    def evaluate(loader: DataLoader) -> float:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for inputs, labels in loader:
                inputs, labels = inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)
                preds = torch.argmax(model(inputs), dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
        return 100.0 * correct / max(total, 1)

    acc_before = evaluate(loaders["test"])
    params_before = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"baseline acc={acc_before:.2f}% params={params_before:,}")

    ranks: str | float = "auto" if args.ranks == "auto" else float(args.ranks)
    cfg = CompressConfig(
        method=args.method,
        layers=lambda name, module: "layer" in name and isinstance(module, nn.Conv2d),
        ranks=ranks,
        use_bn=False,
        finetune=args.ft_epochs > 0,
        finetune_config=FinetuneConfig(
            epochs=max(args.ft_epochs, 1),
            lr=5e-5,
            scheduler="cosine",
            use_amp=(device == "cuda"),
        ),
    )

    result = Compressor(cfg).compress(
        model,
        dataloader={"train": loaders["train"], "val": loaders["val"]},
    )
    acc_after = evaluate(loaders["test"])
    compression_x = params_before / max(result.trainable_params_after, 1)

    from examples.output import make_results_path, save_run_results

    export_path = Path(args.export_path)
    result.export(str(export_path))
    results_path = save_run_results(
        make_results_path("resnet18_hf", output_dir=args.output_dir),
        result=result,
        run={
            "example": "hf_jobs_resnet18",
            "device": device,
            "dataset": args.dataset,
            "method": args.method,
            "ranks": args.ranks,
            "pretrained": args.pretrained,
            "head_epochs": args.head_epochs,
            "ft_epochs": args.ft_epochs,
            "batch_size": args.batch_size,
            "train_size": args.train_size,
            "val_size": args.val_size,
            "test_size": args.test_size,
        },
        metrics={
            "acc_before_pct": round(acc_before, 2),
            "acc_after_pct": round(acc_after, 2),
            "acc_delta_pct": round(acc_after - acc_before, 2),
            "compression_ratio_x": round(compression_x, 2),
        },
        export_path=export_path,
    )
    print(f"Saved run results to {results_path}")

    del model, result
    gc.collect()
    if device == "cuda":
        torch.cuda.empty_cache()

    return {
        "results_path": str(results_path),
        "export_path": str(export_path),
        "acc_before_pct": round(acc_before, 2),
        "acc_after_pct": round(acc_after, 2),
        "compression_ratio_x": round(compression_x, 2),
    }


def push_results(payload: dict, results_path: Path, export_path: Path, output_repo: str) -> None:
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set; skipping Hub upload.")
        return

    from huggingface_hub import HfApi

    api = HfApi(token=token)
    api.create_repo(output_repo, repo_type="model", exist_ok=True)
    if results_path.exists():
        api.upload_file(
            path_or_fileobj=str(results_path),
            path_in_repo=results_path.name,
            repo_id=output_repo,
            repo_type="model",
            commit_message="Add ResNet18 compression results dump",
        )
    if export_path.exists():
        api.upload_file(
            path_or_fileobj=str(export_path),
            path_in_repo=export_path.name,
            repo_id=output_repo,
            repo_type="model",
            commit_message="Add compressed ResNet18 weights",
        )
    print(f"Uploaded results to https://huggingface.co/{output_repo}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorPress ResNet18 HF Jobs runner")
    parser.add_argument("--dataset", default="cifar10")
    parser.add_argument("--method", default="tucker", choices=["tucker", "cpd"])
    parser.add_argument("--ranks", default="auto", help="'auto' or float ratio in (0, 1]")
    parser.add_argument("--head-epochs", type=int, default=5)
    parser.add_argument("--ft-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-size", type=int, default=4096)
    parser.add_argument("--val-size", type=int, default=1024)
    parser.add_argument("--test-size", type=int, default=2048)
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.set_defaults(pretrained=True)
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--export-path", default="compressed_resnet18.pt")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="directory for timestamped JSON result dumps",
    )
    parser.add_argument(
        "--output-repo",
        default="",
        help="Hub model repo for metrics + weights, e.g. username/tensorpress-resnet18-cifar10",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    install_tensorpress()

    payload = run_experiment(args)
    output_repo = args.output_repo or f"{os.environ.get('HF_USER', 'rain92')}/tensorpress-resnet18-{args.dataset}"
    push_results(
        payload,
        Path(payload["results_path"]),
        Path(payload["export_path"]),
        output_repo,
    )


if __name__ == "__main__":
    main()
