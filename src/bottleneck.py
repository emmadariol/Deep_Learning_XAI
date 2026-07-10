"""Concept Bottleneck Model utilities for AwA2."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from src.concepts import AwA2ConceptBank, normalize_class_name
from src.model import freeze_all, unfreeze_modules

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class CBMEpochMetrics:
    epoch: int
    split: str
    loss: float
    class_loss: float
    concept_loss: float
    class_acc: float
    concept_mae: float
    concept_binary_acc: float


@dataclass(frozen=True)
class CBMOutputs:
    class_logits: torch.Tensor
    concept_logits: torch.Tensor
    concept_probs: torch.Tensor


class ConceptTargetDataset(Dataset):
    """Wrap an image manifest dataset with AwA2 class-level concept targets."""

    def __init__(
        self,
        image_dataset: Dataset,
        concept_bank: AwA2ConceptBank,
        concept_indices: list[int],
    ) -> None:
        self.image_dataset = image_dataset
        self.concept_bank = concept_bank
        self.concept_indices = concept_indices
        self.concept_matrix = torch.tensor(
            concept_bank.normalized_matrix()[:, concept_indices],
            dtype=torch.float32,
        )
        self.classes = getattr(image_dataset, "classes", concept_bank.class_names)
        self.samples = getattr(image_dataset, "samples", None)

    def __len__(self) -> int:
        return len(self.image_dataset)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int, str, str]:
        image, label, class_name, filepath = self.image_dataset[index]
        concept_target = self.concept_matrix[int(label)]
        return image, concept_target, int(label), class_name, filepath


class ConceptBottleneckModel(nn.Module):
    """ResNet50 image -> concepts -> class bottleneck model."""

    def __init__(
        self,
        num_classes: int,
        num_concepts: int,
        pretrained: bool = False,
        trainable_backbone_layers: tuple[str, ...] = ("layer4",),
        dropout: float = 0.15,
    ) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)
        feature_dim = backbone.fc.in_features
        backbone.fc = nn.Identity()

        freeze_all(backbone)
        unfreeze_modules(backbone, trainable_backbone_layers)

        self.backbone = backbone
        self.concept_head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feature_dim, num_concepts),
        )
        self.class_head = nn.Linear(num_concepts, num_classes)

    def forward(self, images: torch.Tensor) -> CBMOutputs:
        features = self.backbone(images)
        concept_logits = self.concept_head(features)
        concept_probs = torch.sigmoid(concept_logits)
        class_logits = self.class_head(concept_probs)
        return CBMOutputs(
            class_logits=class_logits,
            concept_logits=concept_logits,
            concept_probs=concept_probs,
        )

    def classify_concepts(self, concept_probs: torch.Tensor) -> torch.Tensor:
        """Classify manually supplied concept probabilities."""
        return self.class_head(concept_probs)


def select_concept_indices(
    concept_bank: AwA2ConceptBank,
    requested_concepts: Iterable[str] | None = None,
    top_k: int = 20,
    always_include: Iterable[str] = ("stripes", "furry", "hooves", "horns", "flippers"),
) -> list[int]:
    """Select a compact concept vocabulary for CBM training."""
    normalized_to_index = {
        normalize_class_name(name): index for index, name in enumerate(concept_bank.concept_names)
    }
    selected: list[int] = []

    def add_name(name: str) -> None:
        key = normalize_class_name(name)
        if key not in normalized_to_index:
            LOGGER.warning("Skipping unknown concept name: %s", name)
            return
        index = normalized_to_index[key]
        if index not in selected:
            selected.append(index)

    for concept_name in requested_concepts or []:
        add_name(concept_name)
    for concept_name in always_include:
        add_name(concept_name)

    matrix = concept_bank.normalized_matrix()
    variances = matrix.var(axis=0)
    for index in np.argsort(variances)[::-1]:
        if int(index) not in selected:
            selected.append(int(index))
        if len(selected) >= top_k:
            break

    return selected[:top_k]


def concept_names_from_indices(concept_bank: AwA2ConceptBank, indices: list[int]) -> list[str]:
    return [concept_bank.concept_names[index] for index in indices]


def load_backbone_checkpoint(
    model: ConceptBottleneckModel,
    checkpoint_path: str | Path,
    device: torch.device,
) -> None:
    """Load a baseline ResNet checkpoint into the CBM backbone when available."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        LOGGER.warning("Backbone checkpoint not found; training CBM backbone from initialization: %s", path)
        return

    checkpoint = torch.load(path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    remapped = {
        f"backbone.{name}": value
        for name, value in state_dict.items()
        if not name.startswith("fc.")
    }
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    LOGGER.info(
        "Loaded CBM backbone from %s with %d missing and %d unexpected keys",
        path,
        len(missing),
        len(unexpected),
    )


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


def concept_binary_accuracy(
    concept_probs: torch.Tensor,
    concept_targets: torch.Tensor,
    threshold: float = 0.5,
) -> float:
    predicted = concept_probs >= threshold
    target = concept_targets >= threshold
    return float((predicted == target).float().mean().item())


def run_cbm_epoch(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    concept_loss_weight: float = 1.0,
    class_loss_weight: float = 1.0,
    max_batches: int | None = None,
) -> CBMEpochMetrics:
    """Run one train/eval pass over a concept-target dataloader."""
    training = optimizer is not None
    model.train(mode=training)

    total_loss = 0.0
    total_class_loss = 0.0
    total_concept_loss = 0.0
    total_correct = 0
    total_samples = 0
    concept_abs_error = 0.0
    concept_binary_correct = 0.0
    concept_values = 0

    for batch_index, batch in enumerate(dataloader):
        if max_batches is not None and batch_index >= max_batches:
            break

        images = batch[0].to(device, non_blocking=True)
        concept_targets = batch[1].to(device, non_blocking=True)
        labels = batch[2].to(device, non_blocking=True)

        if training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(training):
            outputs = model(images)
            concept_loss = F.binary_cross_entropy_with_logits(
                outputs.concept_logits,
                concept_targets,
            )
            class_loss = F.cross_entropy(outputs.class_logits, labels)
            loss = class_loss_weight * class_loss + concept_loss_weight * concept_loss
            if training:
                loss.backward()
                optimizer.step()

        batch_size = labels.size(0)
        total_samples += batch_size
        total_loss += float(loss.detach().item()) * batch_size
        total_class_loss += float(class_loss.detach().item()) * batch_size
        total_concept_loss += float(concept_loss.detach().item()) * batch_size
        total_correct += int((outputs.class_logits.argmax(dim=1) == labels).sum().item())
        concept_abs_error += float(
            torch.abs(outputs.concept_probs.detach() - concept_targets).sum().item()
        )
        concept_binary_correct += float(
            ((outputs.concept_probs.detach() >= 0.5) == (concept_targets >= 0.5))
            .float()
            .sum()
            .item()
        )
        concept_values += concept_targets.numel()

    if total_samples == 0:
        raise RuntimeError("No samples were processed in this CBM epoch.")

    split = "train" if training else "eval"
    return CBMEpochMetrics(
        epoch=0,
        split=split,
        loss=total_loss / total_samples,
        class_loss=total_class_loss / total_samples,
        concept_loss=total_concept_loss / total_samples,
        class_acc=total_correct / total_samples,
        concept_mae=concept_abs_error / max(concept_values, 1),
        concept_binary_acc=concept_binary_correct / max(concept_values, 1),
    )


def train_cbm(
    model: ConceptBottleneckModel,
    dataloaders: dict[str, DataLoader],
    device: torch.device,
    epochs: int,
    optimizer: torch.optim.Optimizer,
    checkpoint_path: str | Path,
    concept_loss_weight: float = 1.0,
    class_loss_weight: float = 1.0,
    max_train_batches: int | None = None,
    max_val_batches: int | None = None,
) -> list[CBMEpochMetrics]:
    """Train CBM and save the checkpoint with the best validation class accuracy."""
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    history: list[CBMEpochMetrics] = []
    best_val_acc = -1.0

    for epoch in range(1, epochs + 1):
        train_metrics = run_cbm_epoch(
            model=model,
            dataloader=dataloaders["train"],
            device=device,
            optimizer=optimizer,
            concept_loss_weight=concept_loss_weight,
            class_loss_weight=class_loss_weight,
            max_batches=max_train_batches,
        )
        val_metrics = run_cbm_epoch(
            model=model,
            dataloader=dataloaders["val"],
            device=device,
            optimizer=None,
            concept_loss_weight=concept_loss_weight,
            class_loss_weight=class_loss_weight,
            max_batches=max_val_batches,
        )
        train_metrics = replace_epoch_and_split(train_metrics, epoch=epoch, split="train")
        val_metrics = replace_epoch_and_split(val_metrics, epoch=epoch, split="val")
        history.extend([train_metrics, val_metrics])

        LOGGER.info(
            "epoch=%03d train_loss=%.4f train_acc=%.4f train_concept_mae=%.4f "
            "val_loss=%.4f val_acc=%.4f val_concept_mae=%.4f",
            epoch,
            train_metrics.loss,
            train_metrics.class_acc,
            train_metrics.concept_mae,
            val_metrics.loss,
            val_metrics.class_acc,
            val_metrics.concept_mae,
        )

        if val_metrics.class_acc > best_val_acc:
            best_val_acc = val_metrics.class_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "epoch": epoch,
                    "val_class_acc": val_metrics.class_acc,
                    "val_concept_mae": val_metrics.concept_mae,
                },
                checkpoint_path,
            )
            LOGGER.info("saved CBM checkpoint: %s", checkpoint_path)

    return history


def replace_epoch_and_split(metrics: CBMEpochMetrics, epoch: int, split: str) -> CBMEpochMetrics:
    return CBMEpochMetrics(
        epoch=epoch,
        split=split,
        loss=metrics.loss,
        class_loss=metrics.class_loss,
        concept_loss=metrics.concept_loss,
        class_acc=metrics.class_acc,
        concept_mae=metrics.concept_mae,
        concept_binary_acc=metrics.concept_binary_acc,
    )


def write_history_csv(history: list[CBMEpochMetrics], output_path: str | Path) -> None:
    path = Path(output_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0].__dict__.keys()))
        writer.writeheader()
        for row in history:
            writer.writerow(row.__dict__)


def per_concept_metrics(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    concept_names: list[str],
    device: torch.device,
    max_batches: int | None = None,
) -> list[dict[str, object]]:
    """Compute per-concept prediction metrics on a dataloader."""
    model.eval()
    predictions: list[torch.Tensor] = []
    targets: list[torch.Tensor] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            concept_targets = batch[1].to(device, non_blocking=True)
            outputs = model(images)
            predictions.append(outputs.concept_probs.detach().cpu())
            targets.append(concept_targets.detach().cpu())

    if not predictions:
        raise RuntimeError("No predictions collected for concept metrics.")

    pred = torch.cat(predictions, dim=0).numpy()
    target = torch.cat(targets, dim=0).numpy()
    rows: list[dict[str, object]] = []
    for index, concept_name in enumerate(concept_names):
        pred_values = pred[:, index]
        target_values = target[:, index]
        pred_centered = pred_values - pred_values.mean()
        target_centered = target_values - target_values.mean()
        denominator = np.linalg.norm(pred_centered) * np.linalg.norm(target_centered)
        pearson = 0.0 if denominator <= 1e-12 else float(np.dot(pred_centered, target_centered) / denominator)
        rows.append(
            {
                "concept": concept_name,
                "mae": float(np.mean(np.abs(pred_values - target_values))),
                "binary_accuracy": float(np.mean((pred_values >= 0.5) == (target_values >= 0.5))),
                "pearson": pearson,
                "target_mean": float(np.mean(target_values)),
                "prediction_mean": float(np.mean(pred_values)),
            }
        )
    return rows


def collect_prediction_rows(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    idx_to_class: dict[int, str],
    concept_names: list[str],
    device: torch.device,
    max_batches: int | None = None,
    top_k_concepts: int = 6,
) -> list[dict[str, object]]:
    """Collect test predictions with top predicted and target concepts."""
    model.eval()
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            concept_targets = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)
            class_names = list(batch[3])
            filepaths = list(batch[4])

            outputs = model(images)
            probs = torch.softmax(outputs.class_logits, dim=1)
            confidences, predictions = probs.max(dim=1)

            for index in range(images.size(0)):
                predicted_concepts = top_concept_string(
                    outputs.concept_probs[index].detach().cpu(),
                    concept_names,
                    top_k=top_k_concepts,
                )
                target_concepts = top_concept_string(
                    concept_targets[index].detach().cpu(),
                    concept_names,
                    top_k=top_k_concepts,
                )
                rows.append(
                    {
                        "filepath": filepaths[index],
                        "true_class": class_names[index],
                        "predicted_class": idx_to_class[int(predictions[index].item())],
                        "correct": bool(predictions[index].item() == labels[index].item()),
                        "confidence": float(confidences[index].item()),
                        "top_predicted_concepts": predicted_concepts,
                        "top_target_concepts": target_concepts,
                    }
                )
    return rows


def top_concept_string(values: torch.Tensor, concept_names: list[str], top_k: int = 6) -> str:
    indices = torch.argsort(values, descending=True)[:top_k].tolist()
    return "; ".join(f"{concept_names[index]}:{float(values[index]):.3f}" for index in indices)


def intervention_rows(
    model: ConceptBottleneckModel,
    concept_bank: AwA2ConceptBank,
    concept_indices: list[int],
    target_classes: list[str],
    device: torch.device,
    top_k_per_class: int = 6,
) -> list[dict[str, object]]:
    """Measure class probability changes when setting one concept to 0 or 1."""
    model.eval()
    concept_names = concept_names_from_indices(concept_bank, concept_indices)
    label_by_class = {
        normalize_class_name(class_name): index
        for index, class_name in enumerate(concept_bank.class_names)
    }
    rows: list[dict[str, object]] = []

    with torch.no_grad():
        for class_name in target_classes:
            key = normalize_class_name(class_name)
            if key not in label_by_class:
                continue
            label = label_by_class[key]
            base = torch.tensor(
                concept_bank.normalized_matrix()[label, concept_indices],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            base_probs = torch.softmax(model.classify_concepts(base), dim=1)
            base_class_prob = float(base_probs[0, label].item())
            candidates: list[dict[str, object]] = []

            for concept_offset, concept_name in enumerate(concept_names):
                zero = base.clone()
                one = base.clone()
                zero[0, concept_offset] = 0.0
                one[0, concept_offset] = 1.0
                zero_prob = float(torch.softmax(model.classify_concepts(zero), dim=1)[0, label].item())
                one_prob = float(torch.softmax(model.classify_concepts(one), dim=1)[0, label].item())
                candidates.append(
                    {
                        "target_class": class_name,
                        "concept": concept_name,
                        "base_class_probability": base_class_prob,
                        "probability_when_concept_zero": zero_prob,
                        "probability_when_concept_one": one_prob,
                        "intervention_delta": one_prob - zero_prob,
                    }
                )

            candidates.sort(key=lambda row: abs(float(row["intervention_delta"])), reverse=True)
            rows.extend(candidates[:top_k_per_class])
    return rows
