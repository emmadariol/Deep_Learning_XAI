"""Training loop utilities for the Oxford-IIIT Pet baseline."""

from __future__ import annotations

import copy
import csv
import logging
import time
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

LOGGER = logging.getLogger(__name__)


@dataclass
class EpochMetrics:
    epoch: int
    train_loss: float
    train_acc: float
    val_loss: float
    val_acc: float
    lr: float


@dataclass
class TrainingResult:
    best_epoch: int
    best_val_acc: float
    best_val_loss: float
    history: list[EpochMetrics]


class EarlyStopping:
    """Early stopping on validation loss."""

    def __init__(self, patience: int, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float("inf")
        self.counter = 0

    def step(self, val_loss: float) -> bool:
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False
        self.counter += 1
        return self.counter >= self.patience


def train_model(
    model: nn.Module,
    dataloaders: dict[str, DataLoader],
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    epochs: int,
    patience: int,
    checkpoint_path: str | Path,
    history_path: str | Path | None = None,
    grad_clip_norm: float | None = None,
) -> TrainingResult:
    """Train a model with validation, early stopping and best-checkpoint saving."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    if history_path is not None:
        history_path = Path(history_path).expanduser().resolve()
        history_path.parent.mkdir(parents=True, exist_ok=True)

    early_stopping = EarlyStopping(patience=patience)
    best_weights = copy.deepcopy(model.state_dict())
    best_epoch = 0
    best_val_acc = 0.0
    best_val_loss = float("inf")
    history: list[EpochMetrics] = []
    first_batch_logged = False
    start_time = time.time()

    for epoch in range(1, epochs + 1):
        train_loss, train_acc, first_batch_logged = run_epoch(
            model=model,
            dataloader=dataloaders["train"],
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            train=True,
            grad_clip_norm=grad_clip_norm,
            log_first_batch=not first_batch_logged,
        )
        val_loss, val_acc, _ = run_epoch(
            model=model,
            dataloader=dataloaders["val"],
            criterion=criterion,
            optimizer=None,
            device=device,
            train=False,
            grad_clip_norm=None,
            log_first_batch=False,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        lr = optimizer.param_groups[0]["lr"]
        metrics = EpochMetrics(
            epoch=epoch,
            train_loss=train_loss,
            train_acc=train_acc,
            val_loss=val_loss,
            val_acc=val_acc,
            lr=lr,
        )
        history.append(metrics)
        LOGGER.info(
            "Epoch %03d/%03d | train_loss=%.4f train_acc=%.4f | val_loss=%.4f val_acc=%.4f | lr=%.6g",
            epoch,
            epochs,
            train_loss,
            train_acc,
            val_loss,
            val_acc,
            lr,
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_acc = val_acc
            best_epoch = epoch
            best_weights = copy.deepcopy(model.state_dict())
            save_checkpoint(
                path=checkpoint_path,
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                val_loss=val_loss,
                val_acc=val_acc,
            )
            LOGGER.info("Saved new best checkpoint: %s", checkpoint_path)

        if early_stopping.step(val_loss):
            LOGGER.info(
                "Early stopping triggered after epoch %d with patience=%d",
                epoch,
                patience,
            )
            break

    model.load_state_dict(best_weights)
    elapsed_minutes = (time.time() - start_time) / 60.0
    LOGGER.info(
        "Training finished in %.2f min | best_epoch=%d best_val_loss=%.4f best_val_acc=%.4f",
        elapsed_minutes,
        best_epoch,
        best_val_loss,
        best_val_acc,
    )

    if history_path is not None:
        write_history(history, history_path)

    return TrainingResult(
        best_epoch=best_epoch,
        best_val_acc=best_val_acc,
        best_val_loss=best_val_loss,
        history=history,
    )


def run_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    train: bool,
    grad_clip_norm: float | None,
    log_first_batch: bool,
) -> tuple[float, float, bool]:
    """Run one train or eval epoch."""
    model.train(mode=train)
    running_loss = 0.0
    running_correct = 0
    running_total = 0
    first_batch_logged = False

    for batch_idx, batch in enumerate(dataloader):
        images, labels = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)

        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(train):
            logits = model(images)
            loss = criterion(logits, labels)
            if train:
                loss.backward()
                if grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
                optimizer.step()

        if log_first_batch and batch_idx == 0:
            LOGGER.info(
                "First train batch tensors: images_shape=%s labels_shape=%s logits_shape=%s images_min=%.4f images_max=%.4f logits_min=%.4f logits_max=%.4f loss=%.4f",
                tuple(images.shape),
                tuple(labels.shape),
                tuple(logits.shape),
                images.min().item(),
                images.max().item(),
                logits.min().item(),
                logits.max().item(),
                loss.item(),
            )
            first_batch_logged = True

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        predictions = logits.argmax(dim=1)
        running_correct += (predictions == labels).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc, first_batch_logged


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    val_loss: float,
    val_acc: float,
) -> None:
    """Save a checkpoint with model and optimizer state."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "val_loss": val_loss,
            "val_acc": val_acc,
        },
        path,
    )


def write_history(history: list[EpochMetrics], path: Path) -> None:
    """Write epoch metrics to CSV."""
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["epoch", "train_loss", "train_acc", "val_loss", "val_acc", "lr"],
        )
        writer.writeheader()
        for row in history:
            writer.writerow(
                {
                    "epoch": row.epoch,
                    "train_loss": row.train_loss,
                    "train_acc": row.train_acc,
                    "val_loss": row.val_loss,
                    "val_acc": row.val_acc,
                    "lr": row.lr,
                }
            )
    LOGGER.info("Wrote training history: %s", path)

