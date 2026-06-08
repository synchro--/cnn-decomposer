"""CP-factorized all-convolutional network (native CPD architecture).

Each logical conv layer is implemented as the four-factor CP chain used in the
research codebase (``CPD_All_Conv`` in ``custom_models.py``):

    1×1 pointwise → k×1 depthwise → 1×k depthwise → 1×1 pointwise

The classifier is also expressed as convolutions (no ``nn.Linear``), so the
entire network is a stack of factorized conv blocks. This is distinct from
TensorPress post-hoc compression: the architecture *is* the decomposition.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class CpFactorConv2d(nn.Module):
    """One convolution expressed as a rank-``R`` CP factor chain."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        rank: int,
        kernel_size: int,
        *,
        padding: int = 0,
    ) -> None:
        super().__init__()
        self.conv_in = nn.Conv2d(in_channels, rank, kernel_size=1)
        self.conv_h = nn.Conv2d(
            rank,
            rank,
            kernel_size=(kernel_size, 1),
            padding=(padding, 0),
            groups=rank,
        )
        self.conv_w = nn.Conv2d(
            rank,
            rank,
            kernel_size=(1, kernel_size),
            padding=(0, padding),
            groups=rank,
        )
        self.conv_out = nn.Conv2d(rank, out_channels, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv_in(x)
        x = self.conv_h(x)
        x = self.conv_w(x)
        return self.conv_out(x)


class CpdAllConvNet(nn.Module):
    """All-conv CP-factorized CNN for 28×28 or 32×32 inputs.

    Channel widths follow the ``CPD_All_Conv`` template (32 → 64 feature maps).
    Ranks ``rank1`` / ``rank2`` / ``rank_fc`` control capacity per stage.

    Parameters
    ----------
    input_channels : int, default=3
        Input channels (grayscale datasets are upsampled to 3 in examples).
    num_classes : int, default=10
        Number of output classes.
    rank1 : int, default=48
        CP rank for stages 1–2.
    rank2 : int, default=48
        CP rank for stages 3–4 and the classifier trunk.
    rank_fc : int, default=48
        CP rank inside the conv classifier head.
    use_bn : bool, default=True
        Insert BatchNorm after each factor conv (research-style).
    """

    def __init__(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        *,
        rank1: int = 48,
        rank2: int = 48,
        rank_fc: int = 48,
        use_bn: bool = True,
    ) -> None:
        super().__init__()
        self.use_bn = use_bn
        c1, c2, c_fc = 32, 64, 512

        # Four CP-factorized spatial stages (each block is one CpFactorConv2d chain).
        self.stage1 = CpFactorConv2d(input_channels, c1, rank1, 3, padding=1)
        self.stage2 = CpFactorConv2d(c1, c1, rank1, 3, padding=0)
        self.stage3 = CpFactorConv2d(c1, c2, rank2, 3, padding=1)
        self.stage4 = CpFactorConv2d(c2, c2, rank2, 3, padding=0)

        self.pool = nn.MaxPool2d(2, 2)
        self.dropout = nn.Dropout2d(0.25)
        self.adaptive_pool = nn.AdaptiveAvgPool2d(1)
        self.relu = nn.ReLU(inplace=True)

        # Classifier as CP-factorized 1×1 convs (spatial size 1×1 after adaptive pool).
        self.cls1 = CpFactorConv2d(c2, c_fc, rank_fc, 1, padding=0)
        self.cls2 = nn.Conv2d(c_fc, num_classes, kernel_size=1)

        if use_bn:
            self.bn1 = nn.BatchNorm2d(c1)
            self.bn2 = nn.BatchNorm2d(c1)
            self.bn3 = nn.BatchNorm2d(c2)
            self.bn4 = nn.BatchNorm2d(c2)
            self.bn_cls = nn.BatchNorm2d(c_fc)

        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def _maybe_bn(self, x: torch.Tensor, bn: nn.BatchNorm2d | None) -> torch.Tensor:
        return bn(x) if self.use_bn and bn is not None else x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self._maybe_bn(self.stage1(x), getattr(self, "bn1", None)))
        x = self.dropout(x)
        x = self.relu(self._maybe_bn(self.stage2(x), getattr(self, "bn2", None)))

        x = self.relu(self._maybe_bn(self.stage3(x), getattr(self, "bn3", None)))
        x = self.pool(x)
        x = self.dropout(x)
        x = self.relu(self._maybe_bn(self.stage4(x), getattr(self, "bn4", None)))
        x = self.pool(x)

        x = self.adaptive_pool(x)
        x = self.relu(self._maybe_bn(self.cls1(x), getattr(self, "bn_cls", None)))
        x = self.cls2(x)
        return x.flatten(1)

    @property
    def conv_parameter_count(self) -> int:
        """All parameters (every layer is a conv)."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    @property
    def fc_parameter_count(self) -> int:
        """Always zero — classifier is conv-only."""
        return 0


def main() -> None:
    """Train ``CpdAllConvNet`` from scratch on a vision dataset."""
    import argparse

    from examples.quickstart import evaluate, get_dataloaders, resolve_device, train

    parser = argparse.ArgumentParser(description="Train native CP all-conv network")
    parser.add_argument("--dataset", default="fashion-mnist")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--train-size", type=int, default=10000)
    parser.add_argument("--test-size", type=int, default=4000)
    args = parser.parse_args()

    device = resolve_device()
    loaders, in_channels, num_classes = get_dataloaders(
        dataset=args.dataset,
        train_size=args.train_size,
        test_size=args.test_size,
    )
    model = CpdAllConvNet(input_channels=in_channels, num_classes=num_classes)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"device={device}  params={n_params:,}", flush=True)

    train(model, loaders, epochs=args.epochs, device=device)
    acc = evaluate(model, loaders["test"], device)
    print(f"test accuracy: {acc:.2f}%", flush=True)


if __name__ == "__main__":
    main()
