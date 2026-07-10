"""TCAV utilities for concept-based explainability on AwA2."""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, Subset

from src.concepts import AwA2ConceptBank, normalize_class_name

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ConceptSampleSelection:
    """Dataset indices used as positive and negative examples for one concept."""

    concept_name: str
    concept_index: int
    positive_indices: list[int]
    negative_indices: list[int]
    positive_threshold: float
    negative_threshold: float
    positive_classes: str
    negative_classes: str


@dataclass(frozen=True)
class TrainedCAV:
    """A trained Concept Activation Vector in raw activation space."""

    concept_name: str
    layer_name: str
    vector: torch.Tensor
    train_accuracy: float
    final_loss: float
    positive_count: int
    negative_count: int
    positive_classes: str
    negative_classes: str


@dataclass(frozen=True)
class TCAVScore:
    """TCAV directional-derivative summary for one class and concept."""

    tcav_score: float
    mean_directional_derivative: float
    std_directional_derivative: float
    n_eval: int


_INDEXED_MODULE_RE = re.compile(r"^(?P<name>[A-Za-z_][A-Za-z0-9_]*)(?:\[(?P<index>-?\d+)\])?$")


def resolve_module(model: nn.Module, layer_name: str) -> nn.Module:
    """Resolve dotted/indexed layer names such as ``layer4`` or ``layer4[-1]``."""
    current: object = model
    for raw_part in layer_name.split("."):
        part = raw_part.strip()
        if not part:
            continue

        if part.lstrip("-").isdigit():
            current = current[int(part)]  # type: ignore[index]
            continue

        match = _INDEXED_MODULE_RE.match(part)
        if match is None:
            raise ValueError(f"Unsupported layer path component: {part!r}")

        name = match.group("name")
        if not hasattr(current, name):
            raise ValueError(f"Module {current.__class__.__name__} has no child {name!r}")
        current = getattr(current, name)

        index = match.group("index")
        if index is not None:
            current = current[int(index)]  # type: ignore[index]

    if not isinstance(current, nn.Module):
        raise ValueError(f"Layer path does not resolve to a torch module: {layer_name}")
    return current


class LayerActivationRecorder:
    """Forward-hook helper that stores one layer output."""

    def __init__(self, model: nn.Module, layer_name: str) -> None:
        self.model = model
        self.layer_name = layer_name
        self.layer = resolve_module(model, layer_name)
        self.activation: torch.Tensor | None = None
        self._handle = self.layer.register_forward_hook(self._save_activation)

    def _save_activation(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor | tuple[torch.Tensor, ...],
    ) -> None:
        if isinstance(output, tuple):
            output = output[0]
        self.activation = output

    def close(self) -> None:
        self._handle.remove()

    def __enter__(self) -> "LayerActivationRecorder":
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def pool_layer_tensor(tensor: torch.Tensor, pool: str = "avg") -> torch.Tensor:
    """Convert layer activations or gradients to one feature vector per image."""
    if tensor.dim() == 2:
        return tensor
    if pool == "flatten":
        return tensor.flatten(start_dim=1)
    if pool != "avg":
        raise ValueError("pool must be 'avg' or 'flatten'.")
    if tensor.dim() == 4:
        return tensor.mean(dim=(2, 3))
    if tensor.dim() == 3:
        return tensor.mean(dim=1)
    return tensor.flatten(start_dim=1)


def find_concept_index(concept_bank: AwA2ConceptBank, concept_name: str) -> int:
    """Return the AwA2 predicate index for a user-provided concept name."""
    normalized_to_index = {
        normalize_class_name(name): index for index, name in enumerate(concept_bank.concept_names)
    }
    key = normalize_class_name(concept_name)
    if key not in normalized_to_index:
        available = ", ".join(concept_bank.concept_names[:20])
        raise ValueError(
            f"Unknown concept {concept_name!r}. "
            f"First available concepts are: {available}"
        )
    return normalized_to_index[key]


def _dataset_samples(dataset: Dataset) -> list[object]:
    samples = getattr(dataset, "samples", None)
    if samples is None:
        raise TypeError("TCAV sample selection expects an ImageManifestDataset with .samples.")
    return samples


def _concept_strengths_for_dataset(
    dataset: Dataset,
    concept_bank: AwA2ConceptBank,
    concept_index: int,
) -> np.ndarray:
    matrix = concept_bank.normalized_matrix()
    samples = _dataset_samples(dataset)
    strengths = []
    for sample in samples:
        strengths.append(float(matrix[int(sample.label), concept_index]))
    return np.array(strengths, dtype=np.float64)


def summarize_classes(dataset: Dataset, indices: list[int], top_k: int = 8) -> str:
    """Return a compact class-count summary for selected dataset indices."""
    samples = _dataset_samples(dataset)
    counts = Counter(str(samples[index].class_name) for index in indices)
    return "; ".join(f"{name}:{count}" for name, count in counts.most_common(top_k))


def select_concept_sample_indices(
    dataset: Dataset,
    concept_bank: AwA2ConceptBank,
    concept_name: str,
    positive_threshold: float = 0.75,
    negative_threshold: float = 0.25,
    max_positive: int = 200,
    max_negative: int = 200,
    min_examples: int = 4,
    seed: int = 42,
    adaptive_thresholds: bool = True,
) -> ConceptSampleSelection:
    """Select positive/negative concept examples using AwA2 attribute strength."""
    if positive_threshold <= negative_threshold:
        raise ValueError("positive_threshold must be greater than negative_threshold.")

    concept_index = find_concept_index(concept_bank, concept_name)
    strengths = _concept_strengths_for_dataset(dataset, concept_bank, concept_index)
    used_positive_threshold = positive_threshold
    used_negative_threshold = negative_threshold

    positive_indices = np.flatnonzero(strengths >= positive_threshold).tolist()
    negative_indices = np.flatnonzero(strengths <= negative_threshold).tolist()

    if adaptive_thresholds and len(positive_indices) < min_examples:
        used_positive_threshold = float(np.quantile(strengths, 0.75))
        positive_indices = np.flatnonzero(strengths >= used_positive_threshold).tolist()
    if adaptive_thresholds and len(negative_indices) < min_examples:
        used_negative_threshold = float(np.quantile(strengths, 0.25))
        negative_indices = np.flatnonzero(strengths <= used_negative_threshold).tolist()

    positive_set = set(positive_indices)
    negative_indices = [index for index in negative_indices if index not in positive_set]

    if len(positive_indices) < min_examples or len(negative_indices) < min_examples:
        raise ValueError(
            f"Not enough examples for concept {concept_name!r}: "
            f"positive={len(positive_indices)} negative={len(negative_indices)}. "
            "Try different thresholds or a larger manifest."
        )

    rng = random.Random(seed)
    rng.shuffle(positive_indices)
    rng.shuffle(negative_indices)
    positive_indices = positive_indices[:max_positive]
    negative_indices = negative_indices[:max_negative]

    return ConceptSampleSelection(
        concept_name=concept_name,
        concept_index=concept_index,
        positive_indices=positive_indices,
        negative_indices=negative_indices,
        positive_threshold=used_positive_threshold,
        negative_threshold=used_negative_threshold,
        positive_classes=summarize_classes(dataset, positive_indices),
        negative_classes=summarize_classes(dataset, negative_indices),
    )


def select_class_sample_indices(
    dataset: Dataset,
    class_name: str,
    max_samples: int = 40,
    seed: int = 42,
) -> list[int]:
    """Select evaluation image indices for one target class."""
    samples = _dataset_samples(dataset)
    target_key = normalize_class_name(class_name)
    indices = [
        index
        for index, sample in enumerate(samples)
        if normalize_class_name(str(sample.class_name)) == target_key
    ]
    rng = random.Random(seed)
    rng.shuffle(indices)
    return indices[:max_samples]


def build_subset_loader(
    dataset: Dataset,
    indices: list[int],
    batch_size: int,
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    """Build a deterministic DataLoader over selected manifest rows."""
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )


def extract_pooled_activations(
    model: nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    device: torch.device,
    pool: str = "avg",
    max_batches: int | None = None,
) -> torch.Tensor:
    """Extract pooled activations from a layer for every image in a dataloader."""
    model.eval()
    activations: list[torch.Tensor] = []

    with LayerActivationRecorder(model, layer_name) as recorder:
        with torch.no_grad():
            for batch_index, batch in enumerate(dataloader):
                if max_batches is not None and batch_index >= max_batches:
                    break
                images = batch[0].to(device, non_blocking=True)
                _ = model(images)
                if recorder.activation is None:
                    raise RuntimeError(f"No activation captured for layer {layer_name}.")
                pooled = pool_layer_tensor(recorder.activation.detach(), pool=pool)
                activations.append(pooled.cpu())

    if not activations:
        raise RuntimeError("No activations were extracted.")
    return torch.cat(activations, dim=0)


def train_cav(
    positive_activations: torch.Tensor,
    negative_activations: torch.Tensor,
    concept_name: str,
    layer_name: str,
    positive_classes: str,
    negative_classes: str,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 1e-4,
    seed: int = 42,
) -> TrainedCAV:
    """Train a linear concept separator and return its raw-space CAV direction."""
    if positive_activations.dim() != 2 or negative_activations.dim() != 2:
        raise ValueError("Activations must be rank-2 tensors [n, features].")
    if positive_activations.size(1) != negative_activations.size(1):
        raise ValueError("Positive and negative activations have different feature sizes.")

    torch.manual_seed(seed)
    x = torch.cat([positive_activations, negative_activations], dim=0).float()
    y = torch.cat(
        [
            torch.ones(positive_activations.size(0)),
            torch.zeros(negative_activations.size(0)),
        ],
        dim=0,
    )
    mean = x.mean(dim=0, keepdim=True)
    std = x.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_standardized = (x - mean) / std

    classifier = nn.Linear(x_standardized.size(1), 1)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    final_loss = 0.0

    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(x_standardized).squeeze(1)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    with torch.no_grad():
        logits = classifier(x_standardized).squeeze(1)
        predictions = (torch.sigmoid(logits) >= 0.5).float()
        train_accuracy = float((predictions == y).float().mean().item())

    standardized_vector = classifier.weight.detach().flatten().cpu()
    raw_vector = standardized_vector / std.flatten().cpu()
    raw_vector = raw_vector / raw_vector.norm().clamp_min(1e-8)

    LOGGER.info(
        "trained CAV concept=%s layer=%s positives=%d negatives=%d acc=%.3f loss=%.4f",
        concept_name,
        layer_name,
        positive_activations.size(0),
        negative_activations.size(0),
        train_accuracy,
        final_loss,
    )

    return TrainedCAV(
        concept_name=concept_name,
        layer_name=layer_name,
        vector=raw_vector,
        train_accuracy=train_accuracy,
        final_loss=final_loss,
        positive_count=int(positive_activations.size(0)),
        negative_count=int(negative_activations.size(0)),
        positive_classes=positive_classes,
        negative_classes=negative_classes,
    )


def compute_tcav_score(
    model: nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    target_label: int,
    cav_vector: torch.Tensor,
    device: torch.device,
    pool: str = "avg",
    max_batches: int | None = None,
) -> TCAVScore:
    """Compute the fraction of positive directional derivatives for a class."""
    model.eval()
    derivatives: list[torch.Tensor] = []

    with LayerActivationRecorder(model, layer_name) as recorder:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break

            images = batch[0].to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(images)
                activation = recorder.activation
                if activation is None:
                    raise RuntimeError(f"No activation captured for layer {layer_name}.")
                if not activation.requires_grad:
                    raise RuntimeError(
                        f"Layer {layer_name!r} activation does not require gradients. "
                        "Use a later layer such as layer4 or unfreeze the selected block."
                    )

                score = logits[:, target_label].sum()
                gradients = torch.autograd.grad(score, activation, retain_graph=False)[0]
                pooled_gradients = pool_layer_tensor(gradients, pool=pool)
                vector = cav_vector.to(device=device, dtype=pooled_gradients.dtype).view(1, -1)
                if pooled_gradients.size(1) != vector.size(1):
                    raise ValueError(
                        f"CAV dimension {vector.size(1)} does not match pooled gradient "
                        f"dimension {pooled_gradients.size(1)}."
                    )
                directional = (pooled_gradients * vector).sum(dim=1)
                derivatives.append(directional.detach().cpu())

    if not derivatives:
        raise RuntimeError("No directional derivatives were computed.")

    values = torch.cat(derivatives, dim=0)
    return TCAVScore(
        tcav_score=float((values > 0).float().mean().item()),
        mean_directional_derivative=float(values.mean().item()),
        std_directional_derivative=float(values.std(unbiased=False).item()),
        n_eval=int(values.numel()),
    )
