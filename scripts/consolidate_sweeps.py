"""Merge sweep JSON files into sweep_results_consolidated.json."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def normalize(row: dict) -> dict:
    """Normalize legacy field names to the consolidated schema."""
    conv_x = row.get("conv_compression_x", row.get("compression_x", 0.0))
    params_before = int(row["params_before"])
    params_after = int(row["params_after"])
    whole_x = row.get(
        "whole_model_compression_x",
        params_before / max(params_after, 1),
    )
    return {
        "dataset": row.get("dataset", "fashion-mnist"),
        "model": row["model"],
        "method": row["method"],
        "compression_ratio": float(row["compression_ratio"]),
        "baseline_epochs": int(row.get("baseline_epochs", 10)),
        "ft_epochs": int(row.get("ft_epochs", 5)),
        "acc_before": float(row["acc_before"]),
        "acc_after": float(row["acc_after"]),
        "acc_delta": float(row.get("acc_delta", row["acc_after"] - row["acc_before"])),
        "params_before": params_before,
        "params_after": params_after,
        "conv_compression_x": float(conv_x),
        "whole_model_compression_x": float(whole_x),
        "param_reduction_pct": float(row["param_reduction_pct"]),
    }


def load_json(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    sources: list[Path] = [
        ROOT / "sweep_results.json",
    ]
    # Historical low-ratio fashion-lenet tucker from git if present.
    hist = ROOT / "sweep_fashion_lenet_tucker_low.json"
    if not hist.exists():
        try:
            raw = subprocess.check_output(
                ["git", "show", "fd89908:sweep_results.json"],
                cwd=ROOT,
                text=True,
            )
            hist.write_text(raw, encoding="utf-8")
        except subprocess.CalledProcessError:
            pass
    if hist.exists():
        sources.append(hist)

    for path in sorted(ROOT.glob("sweep_*.json")):
        if path.name in ("sweep_results_consolidated.json",):
            continue
        sources.append(path)

    seen: set[tuple] = set()
    rows: list[dict] = []
    for path in sources:
        for row in load_json(path):
            norm = normalize(row)
            key = (
                norm["model"],
                norm["method"],
                norm["compression_ratio"],
                norm["baseline_epochs"],
                norm["ft_epochs"],
            )
            if key in seen:
                continue
            seen.add(key)
            rows.append(norm)

    rows.sort(key=lambda r: (r["model"], r["method"], r["compression_ratio"]))
    out = ROOT / "sweep_results_consolidated.json"
    out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"Wrote {len(rows)} rows to {out}")


if __name__ == "__main__":
    main()
