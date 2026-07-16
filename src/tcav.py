"""TCAV utilities for concept-based explainability on AwA2."""

from __future__ import annotations

import logging
import math
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
class CAVDataSplit:
    """Train/validation indices for concept-positive and negative examples."""

    positive_train_indices: list[int]
    positive_validation_indices: list[int]
    negative_train_indices: list[int]
    negative_validation_indices: list[int]
    strategy: str


@dataclass(frozen=True)
class ActivationCache:
    """Deterministic pooled activations aligned with dataset row indices."""

    activations: torch.Tensor
    labels: torch.Tensor
    class_names: list[str]
    layer_name: str
    pool: str


@dataclass(frozen=True)
class TrainedCAV:
    """A trained Concept Activation Vector in raw activation space."""

    concept_name: str
    layer_name: str
    vector: torch.Tensor
    train_accuracy: float
    validation_accuracy: float
    final_loss: float
    validation_loss: float
    positive_count: int
    negative_count: int
    positive_train_count: int
    negative_train_count: int
    positive_validation_count: int
    negative_validation_count: int
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


def _split_group_indices(
    dataset: Dataset,
    indices: list[int],
    validation_fraction: float,
    seed: int,
    prefer_class_disjoint: bool,
) -> tuple[list[int], list[int], str]:
    """Split one CAV side, keeping whole classes apart when possible."""
    if not 0.0 < validation_fraction < 1.0:
        raise ValueError("validation_fraction must be in (0, 1).")
    if len(indices) < 2:
        raise ValueError("At least two indices are required for a CAV split.")

    rng = random.Random(seed)
    samples = _dataset_samples(dataset)
    by_class: dict[int, list[int]] = {}
    for index in indices:
        by_class.setdefault(int(samples[index].label), []).append(index)

    if prefer_class_disjoint and len(by_class) >= 2:
        class_labels = list(by_class)
        rng.shuffle(class_labels)
        target_validation_count = max(1, round(len(indices) * validation_fraction))
        validation_labels: list[int] = []
        validation_count = 0
        for label in class_labels:
            if len(validation_labels) >= len(class_labels) - 1:
                break
            validation_labels.append(label)
            validation_count += len(by_class[label])
            if validation_count >= target_validation_count:
                break
        validation_label_set = set(validation_labels)
        train_indices = [
            index for index in indices if int(samples[index].label) not in validation_label_set
        ]
        validation_indices = [
            index for index in indices if int(samples[index].label) in validation_label_set
        ]
        if train_indices and validation_indices:
            rng.shuffle(train_indices)
            rng.shuffle(validation_indices)
            return train_indices, validation_indices, "class_disjoint"

    shuffled = list(indices)
    rng.shuffle(shuffled)
    validation_count = min(
        len(shuffled) - 1,
        max(1, round(len(shuffled) * validation_fraction)),
    )
    return (
        shuffled[validation_count:],
        shuffled[:validation_count],
        "image_stratified_fallback",
    )


def split_concept_selection(
    dataset: Dataset,
    selection: ConceptSampleSelection,
    validation_fraction: float = 0.25,
    seed: int = 42,
    prefer_class_disjoint: bool = True,
) -> CAVDataSplit:
    """Create a leakage-aware CAV train/validation split."""
    positive_train, positive_validation, positive_strategy = _split_group_indices(
        dataset,
        selection.positive_indices,
        validation_fraction,
        seed,
        prefer_class_disjoint,
    )
    negative_train, negative_validation, negative_strategy = _split_group_indices(
        dataset,
        selection.negative_indices,
        validation_fraction,
        seed + 1,
        prefer_class_disjoint,
    )
    strategy = (
        "class_disjoint"
        if positive_strategy == negative_strategy == "class_disjoint"
        else "mixed_with_image_fallback"
    )
    return CAVDataSplit(
        positive_train_indices=positive_train,
        positive_validation_indices=positive_validation,
        negative_train_indices=negative_train,
        negative_validation_indices=negative_validation,
        strategy=strategy,
    )


def select_random_control_indices(
    dataset_size: int,
    positive_count: int,
    negative_count: int,
    seed: int,
    excluded_indices: set[int] | None = None,
) -> tuple[list[int], list[int]]:
    """Create two disjoint random groups matched to real CAV sample counts."""
    if positive_count < 2 or negative_count < 2:
        raise ValueError("Random control groups require at least two examples per side.")
    excluded = excluded_indices or set()
    available = [index for index in range(dataset_size) if index not in excluded]
    required = positive_count + negative_count
    if required > len(available):
        raise ValueError(
            f"Random controls require {required} rows but only {len(available)} are available."
        )
    rng = random.Random(seed)
    selected = rng.sample(available, required)
    return selected[:positive_count], selected[positive_count:]


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
    excluded_class_names: set[str] | None = None,
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

    excluded_keys = {
        normalize_class_name(name) for name in (excluded_class_names or set())
    }
    if excluded_keys:
        samples = _dataset_samples(dataset)
        positive_indices = [
            index
            for index in positive_indices
            if normalize_class_name(str(samples[index].class_name)) not in excluded_keys
        ]
        negative_indices = [
            index
            for index in negative_indices
            if normalize_class_name(str(samples[index].class_name)) not in excluded_keys
        ]

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


def extract_activation_cache(
    model: nn.Module,
    dataset: Dataset,
    layer_name: str,
    device: torch.device,
    batch_size: int = 16,
    num_workers: int = 0,
    pin_memory: bool = False,
    pool: str = "avg",
) -> ActivationCache:
    """Extract deterministic pooled activations aligned to dataset indices."""
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=False,
    )
    activations = extract_pooled_activations(
        model=model,
        dataloader=loader,
        layer_name=layer_name,
        device=device,
        pool=pool,
    )
    samples = _dataset_samples(dataset)
    if activations.size(0) != len(samples):
        raise RuntimeError("Activation cache is not aligned with dataset rows.")
    labels = torch.tensor([int(sample.label) for sample in samples], dtype=torch.long)
    class_names = [str(sample.class_name) for sample in samples]
    return ActivationCache(
        activations=activations,
        labels=labels,
        class_names=class_names,
        layer_name=layer_name,
        pool=pool,
    )


def extract_pooled_gradients(
    model: nn.Module,
    dataloader: DataLoader,
    layer_name: str,
    target_label: int,
    device: torch.device,
    pool: str = "avg",
    max_batches: int | None = None,
) -> torch.Tensor:
    """Cache target-logit gradients in the selected activation space."""
    model.eval()
    gradients_out: list[torch.Tensor] = []
    with LayerActivationRecorder(model, layer_name) as recorder:
        for batch_index, batch in enumerate(dataloader):
            if max_batches is not None and batch_index >= max_batches:
                break
            images = batch[0].to(device, non_blocking=True)
            model.zero_grad(set_to_none=True)
            with torch.enable_grad():
                logits = model(images)
                activation = recorder.activation
                if activation is None or not activation.requires_grad:
                    raise RuntimeError(
                        f"Layer {layer_name!r} must produce differentiable activations."
                    )
                score = logits[:, target_label].sum()
                gradients = torch.autograd.grad(score, activation, retain_graph=False)[0]
            gradients_out.append(pool_layer_tensor(gradients.detach(), pool=pool).cpu())
    if not gradients_out:
        raise RuntimeError("No target gradients were extracted.")
    return torch.cat(gradients_out, dim=0)


def score_cav_from_gradients(
    pooled_gradients: torch.Tensor,
    cav_vector: torch.Tensor,
) -> TCAVScore:
    """Compute TCAV statistics using cached target gradients."""
    if pooled_gradients.dim() != 2:
        raise ValueError("pooled_gradients must have shape [samples, features].")
    vector = cav_vector.detach().cpu().float().view(1, -1)
    gradients = pooled_gradients.detach().cpu().float()
    if gradients.size(1) != vector.size(1):
        raise ValueError("CAV and gradient feature dimensions do not match.")
    directional = (gradients * vector).sum(dim=1)
    return TCAVScore(
        tcav_score=float((directional > 0).float().mean().item()),
        mean_directional_derivative=float(directional.mean().item()),
        std_directional_derivative=float(directional.std(unbiased=False).item()),
        n_eval=int(directional.numel()),
    )


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
    positive_validation_activations: torch.Tensor | None = None,
    negative_validation_activations: torch.Tensor | None = None,
) -> TrainedCAV:
    """Train and validate a linear separator, then return its raw-space CAV."""
    if positive_activations.dim() != 2 or negative_activations.dim() != 2:
        raise ValueError("Activations must be rank-2 tensors [n, features].")
    if positive_activations.size(1) != negative_activations.size(1):
        raise ValueError("Positive and negative activations have different feature sizes.")

    has_validation = (
        positive_validation_activations is not None
        and negative_validation_activations is not None
    )
    if (positive_validation_activations is None) != (
        negative_validation_activations is None
    ):
        raise ValueError("Both positive and negative validation activations are required.")

    torch.manual_seed(seed)
    x_train = torch.cat([positive_activations, negative_activations], dim=0).float()
    y_train = torch.cat(
        [
            torch.ones(positive_activations.size(0)),
            torch.zeros(negative_activations.size(0)),
        ],
        dim=0,
    )
    mean = x_train.mean(dim=0, keepdim=True)
    std = x_train.std(dim=0, keepdim=True, unbiased=False).clamp_min(1e-6)
    x_train_standardized = (x_train - mean) / std

    if has_validation:
        assert positive_validation_activations is not None
        assert negative_validation_activations is not None
        if positive_validation_activations.size(1) != x_train.size(1) or (
            negative_validation_activations.size(1) != x_train.size(1)
        ):
            raise ValueError("Validation activations have a different feature size.")
        x_validation = torch.cat(
            [positive_validation_activations, negative_validation_activations], dim=0
        ).float()
        y_validation = torch.cat(
            [
                torch.ones(positive_validation_activations.size(0)),
                torch.zeros(negative_validation_activations.size(0)),
            ],
            dim=0,
        )
        x_validation_standardized = (x_validation - mean) / std
    else:
        x_validation_standardized = x_train_standardized
        y_validation = y_train

    classifier = nn.Linear(x_train_standardized.size(1), 1)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()
    final_loss = 0.0

    for _epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        logits = classifier(x_train_standardized).squeeze(1)
        loss = criterion(logits, y_train)
        loss.backward()
        optimizer.step()
        final_loss = float(loss.detach().item())

    with torch.no_grad():
        train_logits = classifier(x_train_standardized).squeeze(1)
        train_predictions = (torch.sigmoid(train_logits) >= 0.5).float()
        train_accuracy = float((train_predictions == y_train).float().mean().item())
        validation_logits = classifier(x_validation_standardized).squeeze(1)
        validation_predictions = (torch.sigmoid(validation_logits) >= 0.5).float()
        validation_accuracy = float(
            (validation_predictions == y_validation).float().mean().item()
        )
        validation_loss = float(criterion(validation_logits, y_validation).item())

    standardized_vector = classifier.weight.detach().flatten().cpu()
    raw_vector = standardized_vector / std.flatten().cpu()
    raw_vector = raw_vector / raw_vector.norm().clamp_min(1e-8)

    LOGGER.info(
        "trained CAV concept=%s layer=%s train_pos=%d train_neg=%d "
        "train_acc=%.3f val_acc=%.3f train_loss=%.4f val_loss=%.4f",
        concept_name,
        layer_name,
        positive_activations.size(0),
        negative_activations.size(0),
        train_accuracy,
        validation_accuracy,
        final_loss,
        validation_loss,
    )

    return TrainedCAV(
        concept_name=concept_name,
        layer_name=layer_name,
        vector=raw_vector,
        train_accuracy=train_accuracy,
        validation_accuracy=validation_accuracy,
        final_loss=final_loss,
        validation_loss=validation_loss,
        positive_count=int(
            positive_activations.size(0)
            + (positive_validation_activations.size(0) if has_validation else 0)
        ),
        negative_count=int(
            negative_activations.size(0)
            + (negative_validation_activations.size(0) if has_validation else 0)
        ),
        positive_train_count=int(positive_activations.size(0)),
        negative_train_count=int(negative_activations.size(0)),
        positive_validation_count=int(
            positive_validation_activations.size(0) if has_validation else 0
        ),
        negative_validation_count=int(
            negative_validation_activations.size(0) if has_validation else 0
        ),
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
    gradients = extract_pooled_gradients(
        model=model,
        dataloader=dataloader,
        layer_name=layer_name,
        target_label=target_label,
        device=device,
        pool=pool,
        max_batches=max_batches,
    )
    return score_cav_from_gradients(gradients, cav_vector)


def paired_permutation_p_value(
    observed_scores: list[float],
    control_scores: list[float],
    seed: int = 42,
    max_permutations: int = 10000,
) -> float:
    """Two-sided paired sign-flip test for real versus random TCAV scores."""
    if len(observed_scores) != len(control_scores) or len(observed_scores) < 2:
        raise ValueError("Paired permutation testing requires equal lists of length >= 2.")
    if max_permutations < 100:
        raise ValueError("max_permutations must be at least 100.")
    differences = np.asarray(observed_scores, dtype=np.float64) - np.asarray(
        control_scores, dtype=np.float64
    )
    if not np.isfinite(differences).all():
        raise ValueError("Permutation inputs must be finite.")
    observed = abs(float(differences.mean()))
    if observed <= 1e-15:
        return 1.0

    exact_count = 2 ** len(differences)
    extreme = 0
    if exact_count <= max_permutations:
        permutation_count = exact_count
        for mask in range(exact_count):
            signs = np.array(
                [1.0 if mask & (1 << index) else -1.0 for index in range(len(differences))]
            )
            extreme += abs(float((differences * signs).mean())) >= observed - 1e-15
    else:
        permutation_count = max_permutations
        rng = np.random.default_rng(seed)
        for _ in range(permutation_count):
            signs = rng.choice(np.array([-1.0, 1.0]), size=len(differences))
            extreme += abs(float((differences * signs).mean())) >= observed - 1e-15
    return float((extreme + 1) / (permutation_count + 1))


def adjust_p_values(
    p_values: list[float],
    method: str = "benjamini_hochberg",
) -> list[float]:
    """Correct a family of p-values with BH-FDR or Bonferroni."""
    if not p_values:
        return []
    values = np.asarray(p_values, dtype=np.float64)
    if not np.isfinite(values).all() or ((values < 0.0) | (values > 1.0)).any():
        raise ValueError("p-values must be finite and in [0, 1].")
    if method == "bonferroni":
        return np.minimum(values * len(values), 1.0).tolist()
    if method != "benjamini_hochberg":
        raise ValueError("method must be benjamini_hochberg or bonferroni.")

    order = np.argsort(values)
    ranked = values[order]
    adjusted_ranked = np.empty_like(ranked)
    running = 1.0
    count = len(values)
    for reverse_index in range(count - 1, -1, -1):
        rank = reverse_index + 1
        running = min(running, ranked[reverse_index] * count / rank)
        adjusted_ranked[reverse_index] = min(running, 1.0)
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = adjusted_ranked
    return adjusted.tolist()


def aggregate_tcav_run_rows(
    run_rows: list[dict[str, object]],
    alpha: float = 0.05,
    correction: str = "benjamini_hochberg",
    permutation_seed: int = 42,
    max_permutations: int = 10000,
) -> list[dict[str, object]]:
    """Aggregate repeated real/random TCAV runs and attach significance."""
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1).")
    grouped: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in run_rows:
        key = (str(row["concept"]), str(row["target_class"]))
        grouped.setdefault(key, []).append(row)

    aggregate: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    for pair_index, ((concept, target_class), rows) in enumerate(sorted(grouped.items())):
        real = [float(row["tcav_score"]) for row in rows]
        random_scores = [float(row["random_tcav_score"]) for row in rows]
        validation = [float(row["cav_validation_accuracy"]) for row in rows]
        random_validation = [
            float(row["random_cav_validation_accuracy"]) for row in rows
        ]
        real_array = np.asarray(real, dtype=np.float64)
        random_array = np.asarray(random_scores, dtype=np.float64)
        effect = real_array - random_array
        run_count = len(rows)
        std = float(real_array.std(ddof=1)) if run_count > 1 else 0.0
        ci_half_width = 1.96 * std / math.sqrt(run_count)
        p_value = paired_permutation_p_value(
            real,
            random_scores,
            seed=permutation_seed + pair_index,
            max_permutations=max_permutations,
        )
        raw_p_values.append(p_value)
        aggregate.append(
            {
                "concept": concept,
                "concept_group": rows[0].get("concept_group", "other"),
                "target_class": target_class,
                "target_label": rows[0]["target_label"],
                "layer": rows[0]["layer"],
                "pool": rows[0]["pool"],
                "runs": run_count,
                "n_eval": rows[0]["n_eval"],
                "cav_validation_accuracy_mean": float(np.mean(validation)),
                "cav_validation_accuracy_std": float(np.std(validation, ddof=1)) if run_count > 1 else 0.0,
                "random_cav_validation_accuracy_mean": float(np.mean(random_validation)),
                "tcav_score": float(real_array.mean()),
                "tcav_std": std,
                "tcav_ci95_low": max(0.0, float(real_array.mean()) - ci_half_width),
                "tcav_ci95_high": min(1.0, float(real_array.mean()) + ci_half_width),
                "random_tcav_mean": float(random_array.mean()),
                "random_tcav_std": float(random_array.std(ddof=1)) if run_count > 1 else 0.0,
                "effect_size": float(effect.mean()),
                "mean_directional_derivative": float(
                    np.mean([float(row["mean_directional_derivative"]) for row in rows])
                ),
                "p_value": p_value,
            }
        )

    corrected = adjust_p_values(raw_p_values, method=correction)
    for row, corrected_p in zip(aggregate, corrected, strict=True):
        row["p_value_corrected"] = corrected_p
        row["correction"] = correction
        row["alpha"] = alpha
        row["significant"] = bool(
            corrected_p < alpha and float(row["effect_size"]) > 0.0
        )
    return aggregate
