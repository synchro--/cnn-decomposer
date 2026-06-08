"""Train dense FashionLeNet vs native CpdAllConvNet from scratch on vision datasets.

Usage
-----
uv run python scripts/benchmark_native_cpd.py
uv run python scripts/benchmark_native_cpd.py --datasets fashion-mnist cifar100
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

from examples.cpd_all_conv import CpdAllConvNet
from examples.fashion_lenet import FashionLeNet
from examples.quickstart import evaluate, get_dataloaders, resolve_device, train


@dataclass
class BenchmarkRow:
    dataset: str
    model: str
    params: int
    conv_params: int
    fc_params: int
    acc_after_train: float


def run_benchmark(
    *,
    dataset: str,
    epochs: int,
    train_size: int,
    test_size: int,
    device: str,
) -> list[BenchmarkRow]:
    loaders, in_channels, num_classes = get_dataloaders(
        dataset=dataset,
        train_size=train_size,
        test_size=test_size,
    )
    rows: list[BenchmarkRow] = []

    models: list[tuple[str, object]] = [
        ("fashion-lenet", FashionLeNet(input_channels=in_channels, num_classes=num_classes)),
        (
            "cpd-all-conv",
            CpdAllConvNet(
                input_channels=in_channels,
                num_classes=num_classes,
                rank1=48,
                rank2=48,
                rank_fc=48,
            ),
        ),
    ]

    for name, model in models:
        print(f"\n>>> {dataset} {name} ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)
        train(model, loaders, epochs=epochs, device=device)
        acc = evaluate(model, loaders["test"], device)
        conv_p = getattr(model, "conv_parameter_count", sum(p.numel() for p in model.parameters()))
        fc_p = getattr(model, "fc_parameter_count", 0)
        rows.append(
            BenchmarkRow(
                dataset=dataset,
                model=name,
                params=sum(p.numel() for p in model.parameters()),
                conv_params=conv_p,
                fc_params=fc_p,
                acc_after_train=acc,
            )
        )
        print(f"    acc={acc:.2f}%", flush=True)

    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark native CPD all-conv vs dense LeNet")
    parser.add_argument("--datasets", nargs="+", default=["fashion-mnist", "cifar100"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--output", default="benchmark_native_cpd.json")
    args = parser.parse_args()

    device = resolve_device()
    print(f"device={device}", flush=True)

    all_rows: list[BenchmarkRow] = []
    for dataset in args.datasets:
        all_rows.extend(
            run_benchmark(
                dataset=dataset,
                epochs=args.epochs,
                train_size=args.train_size,
                test_size=args.test_size,
                device=device,
            )
        )

    out_path = Path(args.output)
    out_path.write_text(json.dumps([asdict(r) for r in all_rows], indent=2) + "\n", encoding="utf-8")
    print(f"\nWrote {len(all_rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
