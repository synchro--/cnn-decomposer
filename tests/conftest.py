"""Shared pytest fixtures for TensorPress.

Fixtures provide small, deterministic modules and tensors so tests stay fast and
run on CPU only (no GPU required).
"""

from __future__ import annotations

import os

import pytest
import torch
import torch.nn as nn


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register optional dataset-backed test flags."""
    parser.addoption(
        "--run-dataset-tests",
        action="store_true",
        default=False,
        help="run tests that download/use torchvision datasets",
    )
    parser.addoption(
        "--dataset-name",
        action="store",
        default="cifar10",
        help="dataset name for optional dataset-backed tests",
    )
    parser.addoption(
        "--dataset-data-dir",
        action="store",
        default="./data",
        help="directory for optional dataset downloads",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers for the test suite."""
    config.addinivalue_line("markers", "dataset: tests that download/use torchvision datasets")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Skip dataset-backed tests unless explicitly enabled."""
    enabled = config.getoption("--run-dataset-tests") or os.getenv("TENSORPRESS_RUN_DATASET_TESTS") == "1"
    if enabled:
        return

    skip_dataset = pytest.mark.skip(
        reason="dataset-backed tests require --run-dataset-tests or TENSORPRESS_RUN_DATASET_TESTS=1"
    )
    for item in items:
        if "dataset" in item.keywords:
            item.add_marker(skip_dataset)


@pytest.fixture
def tiny_conv() -> nn.Conv2d:
    """Single ``Conv2d(16→32, 3×3)`` with random weights.

    Used to check that a factorized replacement preserves output shape and
    behaves sensibly on parameter count without depending on a full model.
    """
    layer = nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False)
    torch.nn.init.normal_(layer.weight)
    return layer


@pytest.fixture
def tiny_model() -> nn.Module:
    """Small module with named conv paths ``blocks.0`` and ``blocks.2``.

    Exercises layer selection by dotted name and per-layer rank dictionaries
    in :class:`~tensorpress.core.compressor.Compressor` tests. ``forward`` is
    defined so the model can be run end-to-end after in-place replacement.
    """

    class M(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.blocks = nn.Sequential(
                nn.Conv2d(3, 8, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.Conv2d(8, 16, kernel_size=3, padding=1),
                nn.ReLU(),
            )

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.blocks(x)

    return M()


@pytest.fixture
def dummy_train_one_batch() -> list[tuple[torch.Tensor, torch.Tensor]]:
    """One (inputs, targets) batch as a one-element list (loader-like).

    Reserved for fine-tuning smoke tests: same layout as iterating a
    ``DataLoader`` without constructing a full dataset.
    """
    inputs = torch.randn(2, 3, 8, 8)
    targets = torch.randint(0, 2, (2,))
    return [(inputs, targets)]
