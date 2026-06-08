"""Sweep compression ratio and epoch settings for demo recommendations.

Usage
-----
.venv/bin/python scripts/sweep_ratio_epochs.py
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass

from examples.quickstart import TinyCNN, evaluate, get_dataloaders, train
from tensorpress import CompressConfig, Compressor
from tensorpress.config import FinetuneConfig


@dataclass
class SweepRow:
    dataset: str
    method: str
    compression_ratio: float | str
    baseline_epochs: int
    ft_epochs: int
    acc_before: float
    acc_after: float
    params_before: int
    params_after: int
    compression_x: float
    param_reduction_pct: float
    acc_delta: float
    train_s: float
    compress_s: float


def run_one(
    *,
    dataset: str,
    method: str,
    compression_ratio: float | str,
    baseline_epochs: int,
    ft_epochs: int,
    device: str,
) -> SweepRow:
    loaders, in_channels, num_classes = get_dataloaders(dataset=dataset)
    model = TinyCNN(input_channels=in_channels, num_classes=num_classes)

    t0 = time.time()
    train(model, loaders, epochs=baseline_epochs, device=device)
    train_s = time.time() - t0

    acc_before = evaluate(model, loaders["test"], device)
    params_before = sum(p.numel() for p in model.parameters())

    auto = compression_ratio == "auto"
    cfg = CompressConfig(
        method=method,
        layers="all",
        ranks="auto",
        compression_ratio=None if auto else float(compression_ratio),
        finetune=ft_epochs > 0,
        finetune_config=FinetuneConfig(
            epochs=max(ft_epochs, 1),
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
    compress_s = time.time() - t1

    acc_after = evaluate(result, loaders["test"], device)
    params_after = result.trainable_params_after
    compression_x = params_before / max(params_after, 1)
    param_reduction_pct = 100.0 * (1.0 - params_after / max(params_before, 1))

    return SweepRow(
        dataset=dataset,
        method=method,
        compression_ratio=compression_ratio,
        baseline_epochs=baseline_epochs,
        ft_epochs=ft_epochs,
        acc_before=acc_before,
        acc_after=acc_after,
        params_before=params_before,
        params_after=params_after,
        compression_x=compression_x,
        param_reduction_pct=param_reduction_pct,
        acc_delta=acc_after - acc_before,
        train_s=train_s,
        compress_s=compress_s,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["cifar10", "fashion-mnist"])
    parser.add_argument("--method", default="tucker", choices=["tucker", "cpd"])
    parser.add_argument(
        "--compression-ratios",
        nargs="+",
        default=["auto", "1.5", "2", "3", "4", "6", "8", "12"],
        help="N-fold size-reduction targets (>= 1), or 'auto' for the VBMF heuristic",
    )
    parser.add_argument("--baseline-epochs", nargs="+", type=int, default=[3, 5, 8])
    parser.add_argument("--ft-epochs", nargs="+", type=int, default=[0, 2, 5])
    args = parser.parse_args()

    import torch

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu" and torch.backends.mps.is_available():
        device = "mps"

    parsed_ratios: list[float | str] = []
    for r in args.compression_ratios:
        parsed_ratios.append("auto" if r == "auto" else float(r))

    rows: list[SweepRow] = []
    for dataset in args.datasets:
        for baseline_epochs in args.baseline_epochs:
            for ft_epochs in args.ft_epochs:
                for compression_ratio in parsed_ratios:
                    print(
                        f"\n>>> {dataset} compression_ratio={compression_ratio} "
                        f"baseline={baseline_epochs} ft={ft_epochs}",
                        flush=True,
                    )
                    row = run_one(
                        dataset=dataset,
                        method=args.method,
                        compression_ratio=compression_ratio,
                        baseline_epochs=baseline_epochs,
                        ft_epochs=ft_epochs,
                        device=device,
                    )
                    rows.append(row)
                    print(
                        f"    before={row.acc_before:.1f}% after={row.acc_after:.1f}% "
                        f"comp={row.compression_x:.2f}x delta={row.acc_delta:+.1f}pp",
                        flush=True,
                    )

    out = [asdict(r) for r in rows]
    path = "sweep_results.json"
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {len(rows)} rows to {path}")


if __name__ == "__main__":
    main()
