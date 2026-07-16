"""Simple training loop for the AwA2 baseline classifier."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.model import set_frozen_batchnorm_eval

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    max_batches: int | None = None,
) -> tuple[float, float]:
    """Run one epoch. Passing an optimizer enables training."""
    training = optimizer is not None
    model.train(mode=training)
    if training:
        set_frozen_batchnorm_eval(model)

    total_loss = 0.0
    total_correct = 0
    total_samples = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch[0].to(device, non_blocking=True)
        labels = batch[1].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)

            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_loss += loss.item() * batch_size
        total_correct += (logits.argmax(dim=1) == labels).sum().item()
        total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("No samples were processed in this epoch.")
    return total_loss / total_samples, total_correct / total_samples


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    checkpoint_path: str | Path,
    history_path: str | Path,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
    early_stopping_patience: int | None = 3,
    early_stopping_min_delta: float = 0.0,
    checkpoint_metadata: dict[str, Any] | None = None,
) -> list[EpochMetrics]:
    """Train, save the best checkpoint and stop after sustained validation stagnation."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    history_path = Path(history_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history_path.parent.mkdir(parents=True, exist_ok=True)
    if early_stopping_patience is not None and early_stopping_patience <= 0:
        raise ValueError("early_stopping_patience must be positive or None.")
    if early_stopping_min_delta < 0.0:
        raise ValueError("early_stopping_min_delta must be non-negative.")

    history: list[EpochMetrics] = []
    best_val_acc = -1.0
    stopping_reference = -1.0
    epochs_without_improvement = 0

    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            max_batches=max_train_batches,
        )
        val_loss, val_acc = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            device=device,
            optimizer=None,
            max_batches=max_val_batches,
        )

        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
        )
        history.append(metrics)
        LOGGER.info(
            "epoch=%03d train_loss=%.4f train_acc=%.4f val_loss=%.4f val_acc=%.4f",
            epoch,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
        )

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_acc": val_acc,
                    "val_loss": val_loss,
                    "metadata": checkpoint_metadata or {},
                },
                checkpoint_path,
            )
            LOGGER.info("saved checkpoint: %s", checkpoint_path)

        if val_acc > stopping_reference + early_stopping_min_delta:
            stopping_reference = val_acc
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            LOGGER.info(
                "early_stopping counter=%d/%s best_reference=%.4f",
                epochs_without_improvement,
                early_stopping_patience,
                stopping_reference,
            )
            if (
                early_stopping_patience is not None
                and epochs_without_improvement >= early_stopping_patience
            ):
                LOGGER.info("early stopping at epoch %d", epoch)
                break

    write_history(history, history_path)
    return history


def write_history(history: list[EpochMetrics], history_path: Path) -> None:
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(row.__dict__)
