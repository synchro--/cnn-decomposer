"""Fashion-MNIST LeNet-style classifier for TensorPress examples.

Adapted from the research ``LenetZhang`` architecture: four convolutional
stages with max pooling, a two-layer FC head, and channel widths widened so
the trainable parameter count exceeds 1M. Only the ``Conv2d`` stages are
candidates for Tucker/CPD compression; the linear classifier stays dense.
"""

from __future__ import annotations

import torch
import torch.nn as nn

ExampleModelName = str


class FashionLeNet(nn.Module):
    """LeNet-style classifier for 28×28 (or 32×32) grayscale/RGB inputs.

    Parameters
    ----------
    input_channels : int, default=3
        Number of input channels (examples upsample grayscale datasets to 3).
    num_classes : int, default=10
        Number of output logits.
    """

    def __init__(self, input_channels: int = 3, num_classes: int = 10) -> None:
        super().__init__()
        # Wider than the original LenetZhang (96/128/256/64) for a richer demo.
        self.conv1 = nn.Conv2d(input_channels, 128, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(128, 192, kernel_size=5, padding=2)
        self.conv3 = nn.Conv2d(192, 384, kernel_size=5, padding=2)
        self.conv4 = nn.Conv2d(384, 128, kernel_size=1)

        self.pool = nn.MaxPool2d(2, 2)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(128 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, num_classes)
        self.relu = nn.ReLU(inplace=True)

        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = self.pool(self.relu(self.conv3(x)))
        x = self.relu(self.conv4(x))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)

    @property
    def conv_parameter_count(self) -> int:
        """Trainable parameters in the four ``Conv2d`` layers only."""
        return sum(
            p.numel() for name, p in self.named_parameters() if name.startswith("conv")
        )


def supported_example_models() -> tuple[ExampleModelName, ...]:
    """Return registered quickstart / sweep model names."""
    return ("tinycnn", "fashion-lenet")


def build_example_model(
    name: ExampleModelName,
    *,
    input_channels: int,
    num_classes: int,
) -> nn.Module:
    """Instantiate a registered example model by name."""
    if name == "tinycnn":
        from examples.quickstart import TinyCNN

        return TinyCNN(input_channels=input_channels, num_classes=num_classes)
    if name == "fashion-lenet":
        return FashionLeNet(input_channels=input_channels, num_classes=num_classes)
    raise ValueError(f"unknown example model: {name!r}; choose from {supported_example_models()}")
