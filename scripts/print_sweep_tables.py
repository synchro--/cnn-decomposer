"""Print markdown tables from sweep_results_consolidated.json."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
rows = json.loads((ROOT / "sweep_results_consolidated.json").read_text(encoding="utf-8"))

HEADER = (
    "| Model | Ratio | Baseline | Compressed | Δ pp | Conv × | Whole × | "
    "Reduction % | Params before | Params after |"
)
SEP = "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"


def print_table(method: str, models: list[str], ratios: list[float]) -> None:
    print(f"\n### {method.upper()}\n")
    print(HEADER)
    print(SEP)
    for model in models:
        for ratio in ratios:
            match = [
                r
                for r in rows
                if r["model"] == model
                and r["method"] == method
                and r["compression_ratio"] == ratio
            ]
            if not match:
                continue
            r = match[0]
            label = model.replace("fashion-lenet-compact", "compact").replace("fashion-lenet", "full")
            print(
                f"| {label} | {ratio:g} | {r['acc_before']:.1f}% | {r['acc_after']:.1f}% | "
                f"{r['acc_delta']:+.1f} | {r['conv_compression_x']:.2f}× | "
                f"{r['whole_model_compression_x']:.2f}× | {r['param_reduction_pct']:.1f}% | "
                f"{r['params_before']:,} | {r['params_after']:,} |"
            )


print_table("tucker", ["tinycnn", "fashion-lenet", "fashion-lenet-compact"],
            [1.5, 2, 3, 4, 6, 8, 10, 15, 20, 25, 30])
print_table("cpd", ["tinycnn"], [8, 10, 15, 20, 25, 30])
