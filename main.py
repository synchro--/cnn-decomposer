"""TensorPress demo: compress a small CNN to a target compression ratio.

This is the top-level entry point for trying TensorPress end-to-end. It
reuses the helpers from :mod:`examples.quickstart` to train a tiny CNN on
a torchvision classification dataset, then runs the canonical five-step
flow against a user-specified compression ratio:

1. build a model
2. train a baseline
3. configure compression (``method`` + desired ``ratio``)
4. compress (and optionally fine-tune)
5. report and export

Usage
-----
python main.py --ratio 0.5
python main.py --method cpd --ratio 0.25 --epochs 2 --ft-epochs 1
python main.py --dataset fashion-mnist --ratio 0.4
"""

from __future__ import annotations

import argparse

from examples.datasets import supported_dataset_names
from examples.output import make_results_path, save_run_results
from examples.quickstart import (
    TinyCNN,
    evaluate,
    get_dataloaders,
    resolve_device,
    train,
)
from tensorpress import CompressConfig, Compressor
from tensorpress.config import FinetuneConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TensorPress compression demo")
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.5,
        help="Target compression ratio in (0, 1]; e.g. 0.5 keeps roughly half the parameters",
    )
    parser.add_argument("--method", default="tucker", choices=["tucker", "cpd"])
    parser.add_argument("--dataset", default="cifar10", choices=supported_dataset_names())
    parser.add_argument("--epochs", type=int, default=1, help="baseline training epochs")
    parser.add_argument("--ft-epochs", type=int, default=1, help="post-compression fine-tune epochs")
    parser.add_argument("--use-bn", action="store_true", help="insert BatchNorm between factor layers")
    parser.add_argument("--export-path", default="compressed_main.pt")
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="directory for timestamped JSON result dumps",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = resolve_device()
    print(
        f"TensorPress demo  device={device}  dataset={args.dataset}  "
        f"method={args.method}  ratio={args.ratio}\n"
    )

    loaders, in_channels, num_classes = get_dataloaders(dataset=args.dataset)

    print(f"Training TinyCNN baseline for {args.epochs} epoch(s)...")
    model = TinyCNN(input_channels=in_channels, num_classes=num_classes)
    train(model, loaders, epochs=args.epochs, device=device)

    acc_before = evaluate(model, loaders["test"], device)
    params_before = sum(p.numel() for p in model.parameters())
    print(f"\nBaseline:   acc={acc_before:.2f}%   params={params_before:,}")

    # The float ``ranks=ratio`` is interpreted as a desired compression ratio
    # (see ``CompressConfig.ranks`` in the docs).
    cfg = CompressConfig(
        method=args.method,
        layers="all",
        ranks=args.ratio,
        use_bn=args.use_bn,
        finetune=args.ft_epochs > 0,
        finetune_config=FinetuneConfig(
            epochs=max(args.ft_epochs, 1),
            lr=1e-4,
            scheduler="cosine",
            use_amp=(device == "cuda"),
        ),
    )

    print(f"\nCompressing with {args.method.upper()} to ratio={args.ratio}...")
    result = Compressor(cfg).compress(
        model,
        dataloader={"train": loaders["train"], "val": loaders["val"]},
    )

    acc_after = evaluate(result, loaders["test"], device)

    print("\nCompression report")
    result.report()
    print(
        f"\nBaseline:   acc={acc_before:.2f}%   params={result.trainable_params_before:,}\n"
        f"Compressed: acc={acc_after:.2f}%   params={result.trainable_params_after:,}\n"
        f"Compression ratio: {result.compression_ratio:.2f}x  "
        f"({result.parameter_reduction_pct:.1f}% fewer parameters)\n"
        f"Accuracy delta:    {acc_after - acc_before:+.2f} percentage points"
    )

    result.export(args.export_path)
    results_path = save_run_results(
        make_results_path("main", output_dir=args.output_dir),
        result=result,
        run={
            "example": "main",
            "device": device,
            "dataset": args.dataset,
            "method": args.method,
            "ratio": args.ratio,
            "epochs": args.epochs,
            "ft_epochs": args.ft_epochs,
            "use_bn": args.use_bn,
        },
        metrics={
            "acc_before_pct": round(acc_before, 2),
            "acc_after_pct": round(acc_after, 2),
            "acc_delta_pct": round(acc_after - acc_before, 2),
            "compression_ratio_x": round(result.compression_ratio, 2),
        },
        export_path=args.export_path,
    )
    print(f"\nSaved compressed model to {args.export_path}")
    print(f"Saved run results to {results_path}")


if __name__ == "__main__":
    main()
