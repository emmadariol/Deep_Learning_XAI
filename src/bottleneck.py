"""Concept Bottleneck Model utilities for AwA2."""

from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50

from src.concepts import AwA2ConceptBank, normalize_class_name
from src.model import freeze_all, set_frozen_batchnorm_eval, unfreeze_modules

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
        self.num_classes = num_classes
        self.num_concepts = num_concepts
        self.backbone_initialization = "imagenet" if pretrained else "random"
        self.trainable_backbone_layers = tuple(trainable_backbone_layers)
        self.dropout = float(dropout)
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
    allow_imagenet_fallback: bool = True,
    expected_class_mapping: dict[int, str] | None = None,
) -> str:
    """Load a trained ResNet backbone without permitting frozen random features.

    If the requested checkpoint is missing, continuing is only valid when the
    model was explicitly initialized with ImageNet weights and that fallback is
    allowed. A randomly initialized, mostly frozen backbone is always rejected.
    """
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        if model.backbone_initialization == "imagenet" and allow_imagenet_fallback:
            LOGGER.warning(
                "Backbone checkpoint not found; using explicit ImageNet initialization: %s",
                path,
            )
            return "imagenet"
        raise FileNotFoundError(
            "CBM backbone checkpoint not found. Supply a trained baseline checkpoint "
            "or pass --use-imagenet-pretrained to use a valid ImageNet fallback. "
            f"Missing path: {path}"
        )

    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a state dictionary: {path}")
    metadata = checkpoint.get("metadata", {})
    if expected_class_mapping is not None and isinstance(metadata, dict):
        stored_mapping = metadata.get("idx_to_class")
        if stored_mapping is None:
            LOGGER.warning(
                "Legacy backbone checkpoint has no class mapping; validating tensor shapes only: %s",
                path,
            )
        else:
            normalized_mapping = {
                int(index): str(name) for index, name in stored_mapping.items()
            }
            if normalized_mapping != expected_class_mapping:
                raise ValueError(
                    "Backbone checkpoint class mapping does not match the CBM dataset."
                )
    remapped = {
        f"backbone.{name.removeprefix('module.')}": value
        for name, value in state_dict.items()
        if not name.removeprefix("module.").startswith("fc.")
    }
    if not remapped:
        raise ValueError(f"No ResNet backbone tensors were found in checkpoint: {path}")
    missing, unexpected = model.load_state_dict(remapped, strict=False)
    LOGGER.info(
        "Loaded CBM backbone from %s with %d missing and %d unexpected keys",
        path,
        len(missing),
        len(unexpected),
    )
    return "awa2_checkpoint"


def trainable_parameters(model: nn.Module):
    return (parameter for parameter in model.parameters() if parameter.requires_grad)


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
    if training:
        set_frozen_batchnorm_eval(model.backbone)

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
    checkpoint_metadata: dict[str, Any] | None = None,
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
                    "model_config": {
                        "architecture": "resnet50_concept_bottleneck",
                        "num_classes": model.num_classes,
                        "num_concepts": model.num_concepts,
                        "trainable_backbone_layers": list(model.trainable_backbone_layers),
                        "dropout": model.dropout,
                        "backbone_initialization": model.backbone_initialization,
                    },
                    "metadata": checkpoint_metadata or {},
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


def concept_confusion_rows(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    concept_names: list[str],
    device: torch.device,
    threshold: float = 0.5,
    max_batches: int | None = None,
) -> list[dict[str, object]]:
    """Compute binary TP/FP/FN/TN counts for each bottleneck concept."""
    model.eval()
    true_positive = torch.zeros(len(concept_names), dtype=torch.long)
    false_positive = torch.zeros(len(concept_names), dtype=torch.long)
    false_negative = torch.zeros(len(concept_names), dtype=torch.long)
    true_negative = torch.zeros(len(concept_names), dtype=torch.long)

    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            concept_targets = batch[1].to(device, non_blocking=True)
            concept_probs = model(images).concept_probs
            predicted = (concept_probs >= threshold).detach().cpu()
            target = (concept_targets >= threshold).detach().cpu()

            true_positive += (predicted & target).sum(dim=0)
            false_positive += (predicted & ~target).sum(dim=0)
            false_negative += (~predicted & target).sum(dim=0)
            true_negative += (~predicted & ~target).sum(dim=0)

    rows: list[dict[str, object]] = []
    for index, concept_name in enumerate(concept_names):
        tp = int(true_positive[index].item())
        fp = int(false_positive[index].item())
        fn = int(false_negative[index].item())
        tn = int(true_negative[index].item())
        total = tp + fp + fn + tn
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        specificity = tn / max(tn + fp, 1)
        f1 = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
        rows.append(
            {
                "concept": concept_name,
                "threshold": threshold,
                "true_positive": tp,
                "false_positive": fp,
                "false_negative": fn,
                "true_negative": tn,
                "support_positive": tp + fn,
                "support_negative": tn + fp,
                "precision": precision,
                "recall": recall,
                "specificity": specificity,
                "f1": f1,
                "accuracy": (tp + tn) / max(total, 1),
            }
        )
    return rows


def collect_prediction_rows(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    idx_to_class: dict[int, str],
    concept_names: list[str],
    device: torch.device,
    baseline_model: nn.Module | None = None,
    max_batches: int | None = None,
    top_k_concepts: int = 6,
) -> list[dict[str, object]]:
    """Collect test predictions with CBM concepts and optional baseline predictions."""
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
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
            if baseline_model is not None:
                baseline_probs = torch.softmax(baseline_model(images), dim=1)
                baseline_confidences, baseline_predictions = baseline_probs.max(dim=1)
            else:
                baseline_confidences = torch.full_like(confidences, float("nan"))
                baseline_predictions = torch.full_like(predictions, -1)

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
                        "cbm_predicted_class": idx_to_class[int(predictions[index].item())],
                        "cbm_correct": bool(predictions[index].item() == labels[index].item()),
                        "cbm_confidence": float(confidences[index].item()),
                        "baseline_predicted_class": (
                            idx_to_class[int(baseline_predictions[index].item())]
                            if int(baseline_predictions[index].item()) >= 0
                            else ""
                        ),
                        "baseline_correct": (
                            bool(baseline_predictions[index].item() == labels[index].item())
                            if int(baseline_predictions[index].item()) >= 0
                            else ""
                        ),
                        "baseline_confidence": (
                            float(baseline_confidences[index].item())
                            if int(baseline_predictions[index].item()) >= 0
                            else ""
                        ),
                        "cbm_agrees_with_baseline": (
                            bool(predictions[index].item() == baseline_predictions[index].item())
                            if int(baseline_predictions[index].item()) >= 0
                            else ""
                        ),
                        "top_predicted_concepts": predicted_concepts,
                        "top_target_concepts": target_concepts,
                    }
                )
    return rows


def collect_cbm_error_rows(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    idx_to_class: dict[int, str],
    concept_names: list[str],
    device: torch.device,
    baseline_model: nn.Module | None = None,
    max_batches: int | None = None,
    top_k_concepts: int = 6,
) -> list[dict[str, object]]:
    """Collect class and concept errors for CBM predictions."""
    model.eval()
    if baseline_model is not None:
        baseline_model.eval()
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
            if baseline_model is not None:
                baseline_probs = torch.softmax(baseline_model(images), dim=1)
                baseline_confidences, baseline_predictions = baseline_probs.max(dim=1)
            else:
                baseline_confidences = torch.full_like(confidences, float("nan"))
                baseline_predictions = torch.full_like(predictions, -1)

            concept_errors = outputs.concept_probs - concept_targets
            abs_errors = concept_errors.abs()

            for index in range(images.size(0)):
                top_error_indices = torch.argsort(abs_errors[index], descending=True)[:top_k_concepts].tolist()
                over_indices = torch.argsort(concept_errors[index], descending=True)[:top_k_concepts].tolist()
                under_indices = torch.argsort(concept_errors[index], descending=False)[:top_k_concepts].tolist()
                cbm_prediction = int(predictions[index].item())
                true_label = int(labels[index].item())
                baseline_prediction = int(baseline_predictions[index].item())
                rows.append(
                    {
                        "filepath": filepaths[index],
                        "true_class": class_names[index],
                        "cbm_predicted_class": idx_to_class[cbm_prediction],
                        "cbm_correct": bool(cbm_prediction == true_label),
                        "cbm_confidence": float(confidences[index].item()),
                        "baseline_predicted_class": (
                            idx_to_class[baseline_prediction] if baseline_prediction >= 0 else ""
                        ),
                        "baseline_correct": (
                            bool(baseline_prediction == true_label) if baseline_prediction >= 0 else ""
                        ),
                        "baseline_confidence": (
                            float(baseline_confidences[index].item()) if baseline_prediction >= 0 else ""
                        ),
                        "cbm_agrees_with_baseline": (
                            bool(cbm_prediction == baseline_prediction) if baseline_prediction >= 0 else ""
                        ),
                        "mean_abs_concept_error": float(abs_errors[index].mean().item()),
                        "max_abs_concept_error": float(abs_errors[index].max().item()),
                        "top_concept_errors": _concept_delta_string(
                            concept_errors[index].detach().cpu(),
                            concept_targets[index].detach().cpu(),
                            outputs.concept_probs[index].detach().cpu(),
                            concept_names,
                            top_error_indices,
                        ),
                        "overpredicted_concepts": _concept_delta_string(
                            concept_errors[index].detach().cpu(),
                            concept_targets[index].detach().cpu(),
                            outputs.concept_probs[index].detach().cpu(),
                            concept_names,
                            over_indices,
                        ),
                        "underpredicted_concepts": _concept_delta_string(
                            concept_errors[index].detach().cpu(),
                            concept_targets[index].detach().cpu(),
                            outputs.concept_probs[index].detach().cpu(),
                            concept_names,
                            under_indices,
                        ),
                    }
                )
    return rows


def summarize_cbm_error_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    """Summarize CBM errors by true/predicted class transition."""
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        if bool(row["cbm_correct"]):
            continue
        key = (str(row["true_class"]), str(row["cbm_predicted_class"]))
        grouped.setdefault(key, []).append(row)

    summary: list[dict[str, object]] = []
    for (true_class, predicted_class), group_rows in sorted(
        grouped.items(),
        key=lambda item: len(item[1]),
        reverse=True,
    ):
        baseline_correct_values = [
            row["baseline_correct"]
            for row in group_rows
            if row["baseline_correct"] != ""
        ]
        baseline_correct_rate = (
            sum(bool(value) for value in baseline_correct_values) / len(baseline_correct_values)
            if baseline_correct_values
            else ""
        )
        summary.append(
            {
                "true_class": true_class,
                "cbm_predicted_class": predicted_class,
                "count": len(group_rows),
                "mean_cbm_confidence": float(np.mean([float(row["cbm_confidence"]) for row in group_rows])),
                "mean_abs_concept_error": float(
                    np.mean([float(row["mean_abs_concept_error"]) for row in group_rows])
                ),
                "mean_max_abs_concept_error": float(
                    np.mean([float(row["max_abs_concept_error"]) for row in group_rows])
                ),
                "baseline_correct_rate": baseline_correct_rate,
            }
        )
    return summary


def _concept_delta_string(
    errors: torch.Tensor,
    targets: torch.Tensor,
    predictions: torch.Tensor,
    concept_names: list[str],
    indices: list[int],
) -> str:
    return "; ".join(
        (
            f"{concept_names[index]}:"
            f"pred={float(predictions[index]):.3f},"
            f"target={float(targets[index]):.3f},"
            f"delta={float(errors[index]):+.3f}"
        )
        for index in indices
    )


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
    """Measure interventions around an AwA2 oracle class prototype.

    These rows do not begin from an image prediction. They answer how the
    learned class head reacts when one coordinate of the class-level attribute
    prototype is toggled, so reports must label them as oracle-prototype tests.
    """
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
                        "intervention_source": "awa2_oracle_class_prototype",
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


def evaluate_oracle_concept_accuracy(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    device: torch.device,
    max_batches: int | None = None,
) -> dict[str, float | int]:
    """Evaluate the class head with ground-truth AwA2 concepts.

    Comparing this oracle accuracy with image-to-concept CBM accuracy separates
    errors in concept prediction from limitations of the concept-to-class head.
    """
    model.eval()
    total = 0
    correct = 0
    with torch.no_grad():
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            concept_targets = batch[1].to(device, non_blocking=True)
            labels = batch[2].to(device, non_blocking=True)
            predictions = model.classify_concepts(concept_targets).argmax(dim=1)
            total += labels.numel()
            correct += int((predictions == labels).sum().item())
    if total == 0:
        raise RuntimeError("No samples processed for oracle concept evaluation.")
    return {
        "oracle_concept_class_accuracy": correct / total,
        "oracle_concept_evaluated_samples": total,
    }


def image_specific_intervention_rows(
    model: ConceptBottleneckModel,
    dataloader: DataLoader,
    idx_to_class: dict[int, str],
    concept_names: list[str],
    device: torch.device,
    max_batches: int | None = None,
    top_k_per_image: int = 3,
    max_images: int | None = 100,
) -> list[dict[str, object]]:
    """Correct predicted concepts to oracle values for individual images.

    Each intervention starts from the image-specific concept vector produced by
    the CBM. One concept is replaced by its AwA2 target value, and the resulting
    change in the true-class probability and final prediction is recorded.
    """
    if top_k_per_image <= 0:
        raise ValueError("top_k_per_image must be positive.")
    model.eval()
    rows: list[dict[str, object]] = []
    processed_images = 0

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
            original_probs = torch.softmax(outputs.class_logits, dim=1)
            original_predictions = original_probs.argmax(dim=1)

            for image_index in range(images.size(0)):
                if max_images is not None and processed_images >= max_images:
                    return rows
                label = int(labels[image_index].item())
                base_concepts = outputs.concept_probs[image_index : image_index + 1]
                target_concepts = concept_targets[image_index]
                base_true_prob = float(original_probs[image_index, label].item())
                base_prediction = int(original_predictions[image_index].item())
                candidates: list[dict[str, object]] = []

                for concept_index, concept_name in enumerate(concept_names):
                    corrected = base_concepts.clone()
                    corrected[0, concept_index] = target_concepts[concept_index]
                    corrected_probs = torch.softmax(model.classify_concepts(corrected), dim=1)[0]
                    corrected_prediction = int(corrected_probs.argmax().item())
                    corrected_true_prob = float(corrected_probs[label].item())
                    candidates.append(
                        {
                            "filepath": filepaths[image_index],
                            "true_class": class_names[image_index],
                            "concept": concept_name,
                            "intervention_source": "image_prediction_to_awa2_oracle_value",
                            "predicted_concept_value": float(base_concepts[0, concept_index].item()),
                            "oracle_concept_value": float(target_concepts[concept_index].item()),
                            "absolute_concept_error": float(
                                abs(base_concepts[0, concept_index] - target_concepts[concept_index]).item()
                            ),
                            "original_predicted_class": idx_to_class[base_prediction],
                            "original_correct": base_prediction == label,
                            "original_true_class_probability": base_true_prob,
                            "intervened_predicted_class": idx_to_class[corrected_prediction],
                            "intervened_correct": corrected_prediction == label,
                            "intervened_true_class_probability": corrected_true_prob,
                            "true_class_probability_delta": corrected_true_prob - base_true_prob,
                        }
                    )

                candidates.sort(
                    key=lambda row: (
                        float(row["true_class_probability_delta"]),
                        float(row["absolute_concept_error"]),
                    ),
                    reverse=True,
                )
                rows.extend(candidates[:top_k_per_image])
                processed_images += 1

    return rows
