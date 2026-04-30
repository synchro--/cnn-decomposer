"""Optional dataset-backed checks for example workflows."""

from __future__ import annotations

import pytest
import torch

from examples.datasets import get_vision_dataloaders, supported_dataset_names
from examples.quickstart import TinyCNN
from tensorpress import CompressConfig, Compressor
from tensorpress.config import FinetuneConfig


@pytest.mark.dataset
def test_selected_dataset_downloads_and_feeds_tinycnn(pytestconfig: pytest.Config) -> None:
    """Download/load the selected dataset and run one compression + fine-tune smoke test."""
    dataset_name = pytestconfig.getoption("--dataset-name")
    data_dir = pytestconfig.getoption("--dataset-data-dir")

    assert dataset_name in supported_dataset_names()
    loaders, spec = get_vision_dataloaders(
        dataset_name,
        data_dir=data_dir,
        batch_size=4,
        train_size=8,
        val_size=4,
        test_size=4,
        download=True,
    )

    inputs, labels = next(iter(loaders["train"]))
    assert inputs.ndim == 4
    assert inputs.shape[1] == spec.channels
    assert labels.numel() == inputs.shape[0]

    model = TinyCNN(input_channels=spec.channels, num_classes=spec.num_classes)
    cfg = CompressConfig(
        method="tucker",
        layers="all",
        ranks={
            "features.0": [4, min(spec.channels, 4)],
            "features.3": [4, 4],
            "features.6": [4, 4],
        },
        finetune=True,
        finetune_config=FinetuneConfig(epochs=1, lr=1e-4, scheduler="none"),
    )

    result = Compressor(cfg).compress(
        model,
        dataloader={"train": loaders["train"], "val": loaders["val"]},
    )
    device = next(result.parameters()).device
    with torch.no_grad():
        logits = result(inputs.to(device))

    assert logits.shape == (inputs.shape[0], spec.num_classes)
    assert result.trainable_params_after < result.trainable_params_before
