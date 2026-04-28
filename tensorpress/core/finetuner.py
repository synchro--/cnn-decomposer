"""Post-compression fine-tuning (pure PyTorch)."""

from __future__ import annotations

import copy
import logging
from typing import Any

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR

from tensorpress.backends.base import BaseBackend
from tensorpress.config.schema import FinetuneConfig

logger = logging.getLogger(__name__)


class FineTuner:
    """Run a short fine-tuning loop after layer replacement."""

    def __init__(self, config: FinetuneConfig, backend: BaseBackend) -> None:
        self.config = config
        self.backend = backend

    def finetune(
        self,
        model: Any,
        dataloader: Any,
        loss_fn: Any | None = None,
    ) -> list[float]:
        """Fine-tune ``model`` using AdamW and an optional LR schedule.

        Parameters
        ----------
        dataloader
            A single training :class:`~torch.utils.data.DataLoader`, or
            ``{\"train\": loader, \"val\": optional_val_loader}``.
        """
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        )
        logger.debug(
            "FineTuner device=%s epochs=%d lr=%g",
            device,
            self.config.epochs,
            self.config.lr,
        )
        model = self.backend.to_device(model, str(device))

        if loss_fn is None:
            loss_fn = nn.CrossEntropyLoss()

        optimizer = optim.AdamW(
            model.parameters(),
            lr=self.config.lr,
            weight_decay=self.config.weight_decay,
        )
        train_loader, val_loader = self._split_loaders(dataloader)
        scheduler = self._build_scheduler(optimizer)

        use_amp = self.config.use_amp and device.type == "cuda"
        scaler = torch.amp.GradScaler("cuda") if use_amp else None

        best_wts = copy.deepcopy(model.state_dict())
        best_acc = 0.0
        history: list[float] = []

        for epoch in range(self.config.epochs):
            model.train()
            running_loss = 0.0

            for batch in train_loader:
                if isinstance(batch, (list, tuple)):
                    inputs, labels = batch[0], batch[1]
                else:
                    raise TypeError("Each batch must be (inputs, labels)")
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad(set_to_none=True)

                if use_amp and scaler is not None:
                    with torch.amp.autocast("cuda"):
                        loss = loss_fn(model(inputs), labels)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss = loss_fn(model(inputs), labels)
                    loss.backward()
                    optimizer.step()

                running_loss += float(loss.item())

            if scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / max(len(train_loader), 1)
            history.append(epoch_loss)
            logger.debug("Epoch %d/%d loss=%.4f", epoch + 1, self.config.epochs, epoch_loss)

            if val_loader is not None:
                acc = self._validate(model, val_loader, device)
                logger.debug("Epoch %d val acc=%.4f", epoch + 1, acc)
                if acc > best_acc:
                    best_acc = acc
                    best_wts = copy.deepcopy(model.state_dict())

        if val_loader is not None:
            model.load_state_dict(best_wts)
        return history

    @staticmethod
    def _split_loaders(dataloader: Any) -> tuple[Any, Any | None]:
        if isinstance(dataloader, dict):
            return dataloader["train"], dataloader.get("val")
        return dataloader, None

    def _build_scheduler(self, optimizer: optim.Optimizer) -> CosineAnnealingLR | StepLR | None:
        name = self.config.scheduler
        if name == "cosine":
            return CosineAnnealingLR(optimizer, T_max=self.config.epochs)
        if name == "step":
            return StepLR(optimizer, step_size=max(1, self.config.epochs // 3), gamma=0.1)
        return None

    @staticmethod
    def _validate(model: nn.Module, val_loader: Any, device: torch.device) -> float:
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                inputs, labels = batch[0], batch[1]
                inputs, labels = inputs.to(device), labels.to(device)
                logits = model(inputs)
                preds = torch.argmax(logits, dim=1)
                correct += int((preds == labels).sum().item())
                total += int(labels.size(0))
        model.train()
        return correct / max(total, 1)
