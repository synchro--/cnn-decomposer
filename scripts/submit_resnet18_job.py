#!/usr/bin/env python3
"""Submit the ResNet18 TensorPress job to Hugging Face Jobs."""

from __future__ import annotations

import sys
from pathlib import Path

from huggingface_hub import get_token, run_uv_job, whoami


def main() -> None:
    script_path = Path(__file__).with_name("hf_jobs_resnet18.py")
    script = script_path.read_text(encoding="utf-8")

    token = get_token()
    if token is None:
        print(
            "No Hugging Face token found. Run:\n"
            "  hf auth login\n"
            "or set HF_TOKEN, then re-run this script.",
            file=sys.stderr,
        )
        sys.exit(1)

    user = whoami(token=token)["name"]
    output_repo = f"{user}/tensorpress-resnet18-cifar10"

    job = run_uv_job(
        script,
        script_args=[
            "--dataset",
            "cifar10",
            "--method",
            "tucker",
            "--compression-ratio",
            "8.0",
            "--head-epochs",
            "5",
            "--ft-epochs",
            "5",
            "--batch-size",
            "32",
            "--train-size",
            "4096",
            "--output-repo",
            output_repo,
        ],
        flavor="t4-small",
        timeout="2h",
        secrets={"HF_TOKEN": "$HF_TOKEN"},
        token=token,
    )

    print(f"Submitted job: {job.id}")
    print(f"Monitor: https://huggingface.co/jobs/{user}/{job.id}")
    print(f"Results will upload to: https://huggingface.co/{output_repo}")


if __name__ == "__main__":
    main()
