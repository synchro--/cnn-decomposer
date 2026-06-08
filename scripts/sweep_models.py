"""Efficient ratio sweep: train once per model, compress at many ratios.

Usage
-----
.venv/bin/python scripts/sweep_models.py --model tinycnn --method tucker
.venv/bin/python scripts/sweep_models.py --model fashion-lenet-compact --method cpd
"""

from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from examples.fashion_lenet import build_example_model, supported_example_models
from examples.quickstart import evaluate, get_dataloaders, train
from tensorpress import CompressConfig, Compressor
from tensorpress.config import FinetuneConfig

TUCKER_RATIOS = [1.5, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30]
CPD_RATIOS = [8, 10, 15, 20, 25, 30]


@dataclass
class SweepRow:
    dataset: str
    model: str
    method: str
    compression_ratio: float
    baseline_epochs: int
    ft_epochs: int
    acc_before: float
    acc_after: float
    params_before: int
    params_after: int
    conv_compression_x: float
    whole_model_compression_x: float
    param_reduction_pct: float
    acc_delta: float
    compress_s: float


def run_sweep(
    *,
    model_name: str,
    method: str,
    ratios: list[float],
    baseline_epochs: int,
    ft_epochs: int,
    device: str,
    train_size: int,
    test_size: int,
) -> list[SweepRow]:
    loaders, in_channels, num_classes = get_dataloaders(
        dataset="fashion-mnist", train_size=train_size, test_size=test_size
    )
    model = build_example_model(
        model_name,
        input_channels=in_channels,
        num_classes=num_classes,
    )

    print(f"Training {model_name} baseline ({baseline_epochs} epochs)...", flush=True)
    train(model, loaders, epochs=baseline_epochs, device=device)
    acc_before = evaluate(model, loaders["test"], device)
    params_before = sum(p.numel() for p in model.parameters())
    print(f"Baseline: acc={acc_before:.1f}% params={params_before:,}", flush=True)

    rows: list[SweepRow] = []
    for ratio in ratios:
        print(f"\n>>> {model_name} {method} ratio={ratio}", flush=True)
        work = copy.deepcopy(model)
        cfg = CompressConfig(
            method=method,
            layers="all",
            ranks="auto",
            compression_ratio=ratio,
            finetune=ft_epochs > 0,
            finetune_config=FinetuneConfig(
                epochs=max(ft_epochs, 1),
                lr=1e-4,
                scheduler="cosine",
                use_amp=(device == "cuda"),
                device=device,
            ),
        )
        t0 = time.time()
        try:
            result = Compressor(cfg).compress(
                work,
                dataloader={"train": loaders["train"], "val": loaders["val"]},
            )
        except Exception as exc:
            print(f"    FAILED: {exc}", flush=True)
            continue
        compress_s = time.time() - t0

        acc_after = evaluate(result, loaders["test"], device)
        params_after = result.trainable_params_after
        whole_x = params_before / max(params_after, 1)
        row = SweepRow(
            dataset="fashion-mnist",
            model=model_name,
            method=method,
            compression_ratio=ratio,
            baseline_epochs=baseline_epochs,
            ft_epochs=ft_epochs,
            acc_before=acc_before,
            acc_after=acc_after,
            params_before=params_before,
            params_after=params_after,
            conv_compression_x=result.subset_compression_ratio,
            whole_model_compression_x=whole_x,
            param_reduction_pct=result.parameter_reduction_pct,
            acc_delta=acc_after - acc_before,
            compress_s=compress_s,
        )
        rows.append(row)
        print(
            f"    acc={acc_after:.1f}% conv={row.conv_compression_x:.2f}x "
            f"whole={row.whole_model_compression_x:.2f}x "
            f"reduction={row.param_reduction_pct:.1f}% delta={row.acc_delta:+.1f}pp",
            flush=True,
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=supported_example_models())
    parser.add_argument("--method", required=True, choices=["tucker", "cpd"])
    parser.add_argument("--compression-ratios", nargs="+", type=float, default=None)
    parser.add_argument("--baseline-epochs", type=int, default=10)
    parser.add_argument("--ft-epochs", type=int, default=5)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=4000)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.compression_ratios is not None:
        ratios = args.compression_ratios
    elif args.method == "tucker":
        ratios = TUCKER_RATIOS
    else:
        ratios = CPD_RATIOS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    # MPS lacks adaptive_avg_pool2d for non-divisible sizes (FashionLeNet 4×4 pool).
    if device == "cpu" and torch.backends.mps.is_available() and args.model == "tinycnn":
        device = "mps"
    print(f"Device: {device}", flush=True)

    rows = run_sweep(
        model_name=args.model,
        method=args.method,
        ratios=ratios,
        baseline_epochs=args.baseline_epochs,
        ft_epochs=args.ft_epochs,
        device=device,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    out_path = args.output or f"sweep_{args.model}_{args.method}.json"
    Path(out_path).write_text(json.dumps([asdict(r) for r in rows], indent=2), encoding="utf-8")
    print(f"\nWrote {len(rows)} rows to {out_path}", flush=True)


if __name__ == "__main__":
    main()
