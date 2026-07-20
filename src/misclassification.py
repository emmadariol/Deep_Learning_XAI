"""Contrastive diagnostics for image-classification errors.

The functions in this module compare the class selected by a classifier with
the ground-truth class on the same image. They deliberately separate model
scores, spatial attribution and concept-bottleneck evidence so that each
quantity keeps a precise interpretation.
"""

from __future__ import annotations

import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

_CACHE_ROOT = Path(tempfile.gettempdir()) / "deep_learning_xai"
_MPLCONFIGDIR = _CACHE_ROOT / "matplotlib"
_XDG_CACHE_HOME = _CACHE_ROOT / "xdg-cache"
_MPLCONFIGDIR.mkdir(parents=True, exist_ok=True)
_XDG_CACHE_HOME.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))
os.environ.setdefault("XDG_CACHE_HOME", str(_XDG_CACHE_HOME))

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import denormalize_batch
from src.explainability_audit import rank_correlation, topk_iou
from src.xai import blurred_baseline, overlay_heatmap


@dataclass(frozen=True)
class TargetPairScores:
    """Scores for fixed true and wrong target classes."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    true_logits: torch.Tensor
    wrong_logits: torch.Tensor
    true_probabilities: torch.Tensor
    wrong_probabilities: torch.Tensor
    margins: torch.Tensor


def score_target_pair(
    logits: torch.Tensor,
    true_targets: torch.Tensor,
    wrong_targets: torch.Tensor,
) -> TargetPairScores:
    """Return true/wrong scores and ``wrong - true`` decision margins."""
    if logits.dim() != 2:
        raise ValueError("logits must have shape [B, C].")
    batch_size, class_count = logits.shape
    for name, targets in (("true_targets", true_targets), ("wrong_targets", wrong_targets)):
        if targets.dim() != 1 or targets.size(0) != batch_size:
            raise ValueError(f"{name} must have shape [B].")
        if targets.dtype == torch.bool or targets.dtype.is_floating_point:
            raise ValueError(f"{name} must contain integer class indices.")
        if targets.numel() and ((targets < 0).any() or (targets >= class_count).any()):
            raise ValueError(f"{name} contains an out-of-range class index.")

    true_targets = true_targets.to(device=logits.device, dtype=torch.long)
    wrong_targets = wrong_targets.to(device=logits.device, dtype=torch.long)
    probabilities = torch.softmax(logits, dim=1)
    true_logits = logits.gather(1, true_targets[:, None]).squeeze(1)
    wrong_logits = logits.gather(1, wrong_targets[:, None]).squeeze(1)
    true_probabilities = probabilities.gather(1, true_targets[:, None]).squeeze(1)
    wrong_probabilities = probabilities.gather(1, wrong_targets[:, None]).squeeze(1)
    return TargetPairScores(
        logits=logits,
        probabilities=probabilities,
        true_logits=true_logits,
        wrong_logits=wrong_logits,
        true_probabilities=true_probabilities,
        wrong_probabilities=wrong_probabilities,
        margins=wrong_logits - true_logits,
    )


def saliency_pair_diagnostics(
    first: torch.Tensor,
    second: torch.Tensor,
    top_fraction: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-image top-k IoU and Spearman correlation for two map batches."""
    if first.shape != second.shape or first.dim() != 4 or first.size(1) != 1:
        raise ValueError("saliency maps must share shape [B, 1, H, W].")
    ious = [topk_iou(first[index], second[index], top_fraction) for index in range(first.size(0))]
    correlations = [rank_correlation(first[index], second[index]) for index in range(first.size(0))]
    return (
        torch.tensor(ious, dtype=torch.float32),
        torch.tensor(correlations, dtype=torch.float32),
    )


def background_saliency_fraction(
    maps: torch.Tensor,
    background_mask: torch.Tensor,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Return the fraction of non-negative saliency assigned to background."""
    if maps.dim() != 4 or maps.size(1) != 1:
        raise ValueError("maps must have shape [B, 1, H, W].")
    if background_mask.shape != maps.shape:
        raise ValueError("background_mask must have the same shape as maps.")
    nonnegative = maps.clamp_min(0).float()
    total = nonnegative.flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    background = (
        nonnegative * background_mask.to(device=maps.device, dtype=nonnegative.dtype)
    ).flatten(start_dim=1).sum(dim=1)
    return background / total


def replace_top_salient_pixels(
    inputs: torch.Tensor,
    maps: torch.Tensor,
    fraction: float,
    baseline: torch.Tensor | None = None,
) -> torch.Tensor:
    """Replace exactly the most salient spatial fraction with a blurred baseline."""
    if not math.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be a finite value in [0, 1].")
    if inputs.dim() != 4 or maps.shape != (inputs.size(0), 1, inputs.size(2), inputs.size(3)):
        raise ValueError("inputs and maps must have compatible [B, C, H, W] shapes.")
    if baseline is None:
        baseline = blurred_baseline(inputs)
    if baseline.shape != inputs.shape:
        raise ValueError("baseline must have the same shape as inputs.")

    pixel_count = maps.size(2) * maps.size(3)
    selected_count = int(math.ceil(fraction * pixel_count))
    if selected_count == 0:
        spatial_mask = torch.zeros_like(maps, dtype=torch.bool)
    elif selected_count >= pixel_count:
        spatial_mask = torch.ones_like(maps, dtype=torch.bool)
    else:
        flat_maps = maps.detach().flatten(start_dim=1)
        top_indices = torch.topk(flat_maps, k=selected_count, dim=1).indices
        flat_mask = torch.zeros_like(flat_maps, dtype=torch.bool)
        flat_mask.scatter_(1, top_indices, True)
        spatial_mask = flat_mask.view_as(maps)
    return torch.where(spatial_mask.expand_as(inputs), baseline, inputs)


def concept_evidence_rows(
    class_head: nn.Linear,
    concept_probabilities: torch.Tensor,
    true_prototypes: torch.Tensor,
    wrong_prototypes: torch.Tensor,
    true_targets: torch.Tensor,
    wrong_targets: torch.Tensor,
    concept_names: list[str],
) -> list[dict[str, float | int | str]]:
    """Decompose a CBM's wrong-vs-true margin into concept contributions.

    This is evidence for the concept bottleneck model, not a causal explanation
    of a separate end-to-end classifier.
    """
    if not isinstance(class_head, nn.Linear):
        raise TypeError("class_head must be a linear layer for exact decomposition.")
    if concept_probabilities.dim() != 2:
        raise ValueError("concept_probabilities must have shape [B, K].")
    if true_prototypes.shape != concept_probabilities.shape:
        raise ValueError("true_prototypes must match concept_probabilities.")
    if wrong_prototypes.shape != concept_probabilities.shape:
        raise ValueError("wrong_prototypes must match concept_probabilities.")
    if concept_probabilities.size(1) != len(concept_names):
        raise ValueError("concept_names does not match the concept dimension.")

    true_targets = true_targets.to(device=concept_probabilities.device, dtype=torch.long)
    wrong_targets = wrong_targets.to(device=concept_probabilities.device, dtype=torch.long)
    with torch.no_grad():
        logits = class_head(concept_probabilities)
        scores = score_target_pair(logits, true_targets, wrong_targets)
        weight_delta = class_head.weight[wrong_targets] - class_head.weight[true_targets]
        contributions = weight_delta * concept_probabilities
        if class_head.bias is None:
            bias_delta = torch.zeros(
                concept_probabilities.size(0),
                dtype=concept_probabilities.dtype,
                device=concept_probabilities.device,
            )
        else:
            bias_delta = class_head.bias[wrong_targets] - class_head.bias[true_targets]
        reconstructed_margins = contributions.sum(dim=1) + bias_delta

    rows: list[dict[str, float | int | str]] = []
    for image_index in range(concept_probabilities.size(0)):
        base_concepts = concept_probabilities[image_index]
        for concept_index, concept_name in enumerate(concept_names):
            corrected = base_concepts.clone()
            corrected[concept_index] = true_prototypes[image_index, concept_index]
            with torch.no_grad():
                corrected_logits = class_head(corrected[None])
                corrected_scores = score_target_pair(
                    corrected_logits,
                    true_targets[image_index : image_index + 1],
                    wrong_targets[image_index : image_index + 1],
                )
                corrected_probabilities = torch.softmax(corrected_logits, dim=1)
                corrected_confidence, corrected_prediction = corrected_probabilities.max(dim=1)
            predicted_value = float(base_concepts[concept_index].item())
            true_value = float(true_prototypes[image_index, concept_index].item())
            wrong_value = float(wrong_prototypes[image_index, concept_index].item())
            rows.append(
                {
                    "image_index": image_index,
                    "concept_index": concept_index,
                    "concept": concept_name,
                    "predicted_value": predicted_value,
                    "true_prototype_value": true_value,
                    "wrong_prototype_value": wrong_value,
                    "distance_to_true_prototype": abs(predicted_value - true_value),
                    "distance_to_wrong_prototype": abs(predicted_value - wrong_value),
                    "prototype_alignment_gap": (
                        abs(predicted_value - true_value) - abs(predicted_value - wrong_value)
                    ),
                    "wrong_vs_true_weight": float(weight_delta[image_index, concept_index].item()),
                    "wrong_vs_true_margin_contribution": float(
                        contributions[image_index, concept_index].item()
                    ),
                    "wrong_vs_true_bias": float(bias_delta[image_index].item()),
                    "original_cbm_margin": float(scores.margins[image_index].item()),
                    "reconstructed_cbm_margin": float(
                        reconstructed_margins[image_index].item()
                    ),
                    "margin_reconstruction_error": float(
                        reconstructed_margins[image_index].item()
                        - scores.margins[image_index].item()
                    ),
                    "corrected_cbm_margin": float(corrected_scores.margins[0].item()),
                    "correction_margin_delta": float(
                        corrected_scores.margins[0].item() - scores.margins[image_index].item()
                    ),
                    "correction_true_probability_delta": float(
                        corrected_scores.true_probabilities[0].item()
                        - scores.true_probabilities[image_index].item()
                    ),
                    "correction_wrong_probability_delta": float(
                        corrected_scores.wrong_probabilities[0].item()
                        - scores.wrong_probabilities[image_index].item()
                    ),
                    "corrected_prediction_index": int(corrected_prediction[0].item()),
                    "corrected_prediction_confidence": float(
                        corrected_confidence[0].item()
                    ),
                }
            )
    return rows


def _as_numpy_image(images: torch.Tensor, index: int) -> np.ndarray:
    denormalized = denormalize_batch(images.detach().cpu()).clamp(0.0, 1.0)
    return denormalized[index].permute(1, 2, 0).numpy()


def save_contrastive_attribution_figure(
    images: torch.Tensor,
    wrong_maps: torch.Tensor,
    true_maps: torch.Tensor,
    scores: TargetPairScores,
    true_names: list[str],
    wrong_names: list[str],
    method: str,
    output_path: str | Path,
    background_mask: torch.Tensor | None = None,
    top_fraction: float = 0.2,
    actual_names: list[str] | None = None,
) -> None:
    """Save target-contrast maps with per-example error diagnostics."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    row_count = images.size(0)
    if actual_names is None:
        actual_names = wrong_names
    if len(actual_names) != row_count:
        raise ValueError("actual_names must match the number of images.")
    target_iou, target_spearman = saliency_pair_diagnostics(wrong_maps, true_maps, top_fraction)
    if background_mask is not None:
        wrong_background = background_saliency_fraction(wrong_maps, background_mask)
        true_background = background_saliency_fraction(true_maps, background_mask)
    else:
        wrong_background = torch.full((row_count,), float("nan"))
        true_background = torch.full((row_count,), float("nan"))

    top_percent = int(round(top_fraction * 100))
    fig, axes = plt.subplots(row_count, 6, figsize=(22.0, 3.65 * row_count), squeeze=False)
    for index in range(row_count):
        image = _as_numpy_image(images, index)
        wrong_map = wrong_maps[index, 0].detach().cpu().numpy()
        true_map = true_maps[index, 0].detach().cpu().numpy()
        fixed_target = actual_names[index] != wrong_names[index]
        contrast_label = "contrast target"
        metric_prefix = "contrast - true"

        axes[index, 0].imshow(image)
        axes[index, 0].set_title(
            f"original image\ntrue: {true_names[index]}\nactual pred: {actual_names[index]}\n"
            f"{contrast_label}: {wrong_names[index]}\n"
            f"p({wrong_names[index]})={scores.wrong_probabilities[index]:.3f}, "
            f"p(true)={scores.true_probabilities[index]:.3f}"
        )
        axes[index, 1].imshow(overlay_heatmap(image, wrong_map))
        axes[index, 1].set_title(f"explained target: {contrast_label}\n{wrong_names[index]}")
        axes[index, 2].imshow(overlay_heatmap(image, true_map))
        axes[index, 2].set_title(f"explained target: ground truth\n{true_names[index]}")
        axes[index, 3].imshow(wrong_map, cmap="magma", vmin=0.0, vmax=1.0)
        axes[index, 3].set_title(f"raw heatmap\n{contrast_label}")
        axes[index, 4].imshow(true_map, cmap="magma", vmin=0.0, vmax=1.0)
        axes[index, 4].set_title("raw heatmap\ntrue target")

        probability_gap = (
            scores.wrong_probabilities[index] - scores.true_probabilities[index]
        ).item()
        background_gap = (wrong_background[index] - true_background[index]).item()
        metrics_text = (
            "Contrast diagnostics\n\n"
            f"{metric_prefix} logit: {scores.margins[index]:+.3f}\n"
            f"{metric_prefix} prob.: {probability_gap:+.3f}\n\n"
            f"map IoU@top{top_percent}: {target_iou[index]:.3f}\n"
            f"map Spearman: {target_spearman[index]:.3f}\n\n"
            f"background saliency\n"
            f"{contrast_label}: {wrong_background[index]:.3f}\n"
            f"true target: {true_background[index]:.3f}\n"
            f"{metric_prefix}: {background_gap:+.3f}"
        )
        axes[index, 5].text(
            0.03,
            0.97,
            metrics_text,
            transform=axes[index, 5].transAxes,
            va="top",
            ha="left",
            fontsize=9.2,
            family="monospace",
            bbox={
                "boxstyle": "round,pad=0.55",
                "facecolor": "#f8fafc",
                "edgecolor": "#cbd5e1",
                "linewidth": 0.9,
            },
        )
        axes[index, 5].set_title("local metrics")
        for axis in axes[index]:
            axis.axis("off")
    method_label = method.replace("_", " ").title()
    if any(actual != wrong for actual, wrong in zip(actual_names, wrong_names, strict=True)):
        title = f"Fixed target contrast: {method_label}"
    else:
        title = f"Misclassification target contrast: {method_label}"
    fig.suptitle(title, fontsize=14, y=1.002)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_perturbation_margin_figure(
    images: torch.Tensor,
    scores_by_condition: dict[str, TargetPairScores],
    predictions_by_condition: dict[str, torch.Tensor],
    idx_to_class: dict[int, str],
    true_names: list[str],
    wrong_names: list[str],
    output_path: str | Path,
) -> None:
    """Plot probability and margin trajectories under controlled perturbations."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    conditions = list(scores_by_condition)
    positions = np.arange(len(conditions))
    row_count = images.size(0)
    fig, axes = plt.subplots(row_count, 3, figsize=(15.0, 3.45 * row_count), squeeze=False)

    for index in range(row_count):
        axes[index, 0].imshow(_as_numpy_image(images, index))
        axes[index, 0].set_title(f"{true_names[index]} -> {wrong_names[index]}")
        axes[index, 0].axis("off")

        wrong_values = [
            float(scores_by_condition[name].wrong_probabilities[index].item())
            for name in conditions
        ]
        true_values = [
            float(scores_by_condition[name].true_probabilities[index].item())
            for name in conditions
        ]
        axes[index, 1].plot(positions, wrong_values, "o-", color="#b91c1c", label="wrong target")
        axes[index, 1].plot(positions, true_values, "o-", color="#0f766e", label="true target")
        probability_ceiling = max(0.12, min(1.0, max(wrong_values + true_values) * 1.28))
        axes[index, 1].set_ylim(0.0, probability_ceiling)
        axes[index, 1].set_ylabel("softmax probability")
        axes[index, 1].set_xticks(positions, [name.replace("_", "\n") for name in conditions])
        axes[index, 1].grid(alpha=0.25)
        axes[index, 1].legend(loc="best", fontsize=8)

        margins = [float(scores_by_condition[name].margins[index].item()) for name in conditions]
        colors = ["#b91c1c" if value > 0 else "#0f766e" for value in margins]
        bars = axes[index, 2].bar(positions, margins, color=colors)
        axes[index, 2].axhline(0.0, color="black", linewidth=0.9)
        axes[index, 2].set_ylabel("logit margin: wrong - true")
        axes[index, 2].set_xticks(positions, [name.replace("_", "\n") for name in conditions])
        axes[index, 2].grid(axis="y", alpha=0.25)
        lower = min(0.0, min(margins))
        upper = max(0.0, max(margins))
        span = max(upper - lower, 0.1)
        axes[index, 2].set_ylim(lower - 0.18 * span, upper + 0.28 * span)
        for condition, bar in zip(conditions, bars, strict=True):
            predicted_label = int(predictions_by_condition[condition][index].item())
            axes[index, 2].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height(),
                idx_to_class[predicted_label],
                ha="center",
                va="bottom" if bar.get_height() >= 0 else "top",
                rotation=90,
                fontsize=7,
            )

    fig.suptitle("Decision response on misclassified images", fontsize=14, y=1.002)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_deletion_curves_figure(
    rows: list[dict[str, object]],
    methods: Iterable[str],
    output_path: str | Path,
) -> None:
    """Save aggregate true/wrong probability curves for target-ranked deletion."""
    methods = list(methods)
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(len(methods), 2, figsize=(12.5, 4.2 * len(methods)), squeeze=False)
    for method_index, method in enumerate(methods):
        for role_index, role in enumerate(("wrong", "true")):
            axis = axes[method_index, role_index]
            subset = [
                row for row in rows
                if row["xai_method"] == method and row["ranking_target"] == role
            ]
            fractions = sorted({float(row["deleted_fraction"]) for row in subset})
            wrong_means = [
                float(np.mean([
                    float(row["wrong_probability"])
                    for row in subset
                    if float(row["deleted_fraction"]) == fraction
                ]))
                for fraction in fractions
            ]
            true_means = [
                float(np.mean([
                    float(row["true_probability"])
                    for row in subset
                    if float(row["deleted_fraction"]) == fraction
                ]))
                for fraction in fractions
            ]
            axis.plot(fractions, wrong_means, "o-", color="#b91c1c", label="wrong target")
            axis.plot(fractions, true_means, "o-", color="#0f766e", label="true target")
            method_label = method.replace("_", " ").title()
            axis.set_title(f"{method_label}: pixels ranked for {role} target")
            axis.set_xlabel("fraction replaced by blurred baseline")
            axis.set_ylabel("mean class probability")
            probability_ceiling = max(0.12, min(1.0, max(wrong_means + true_means) * 1.28))
            axis.set_ylim(0.0, probability_ceiling)
            axis.grid(alpha=0.25)
            axis.legend()
    fig.suptitle("Deletion test on misclassified images", fontsize=14, y=1.002)
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_concept_evidence_figure(
    rows: list[dict[str, object]],
    image_summaries: list[dict[str, object]],
    output_path: str | Path,
    top_k: int = 8,
) -> None:
    """Plot the strongest CBM concept contributions for each baseline error."""
    if not rows:
        return
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(
        len(image_summaries),
        1,
        figsize=(12.0, 3.4 * len(image_summaries)),
        squeeze=False,
    )
    for image_index, summary in enumerate(image_summaries):
        axis = axes[image_index, 0]
        candidates = [row for row in rows if int(row["image_index"]) == image_index]
        selected = sorted(
            candidates,
            key=lambda row: abs(float(row["wrong_vs_true_margin_contribution"])),
            reverse=True,
        )[:top_k]
        selected.reverse()
        labels = [str(row["concept"]) for row in selected]
        values = [float(row["wrong_vs_true_margin_contribution"]) for row in selected]
        colors = ["#b91c1c" if value > 0 else "#0f766e" for value in values]
        axis.barh(labels, values, color=colors)
        axis.axvline(0.0, color="black", linewidth=0.9)
        axis.grid(axis="x", alpha=0.25)
        axis.set_xlabel("CBM contribution to wrong - true logit margin")
        axis.set_title(
            f"baseline: {summary['true_class']} -> {summary['wrong_class']} | "
            f"CBM top-1: {summary['cbm_predicted_class']} | "
            f"CBM margin={float(summary['cbm_wrong_vs_true_margin']):+.3f}"
        )
    fig.suptitle(
        "Concept bottleneck comparator (evidence for the CBM, not the direct ResNet)",
        fontsize=13,
        y=1.002,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def save_cbm_error_decomposition_figure(
    image: torch.Tensor,
    rows: list[dict[str, object]],
    summary: dict[str, object],
    output_path: str | Path,
    top_k: int = 8,
) -> None:
    """Save a four-panel semantic decomposition of one wrong CBM prediction.

    The panels connect the input image, its predicted-versus-AwA2 concept
    values, the exact linear-head contributions to the predicted-versus-true
    logit margin, and one-concept oracle corrections.  Positive contribution
    bars support the wrong prediction; negative bars support the true class.
    """
    if not rows:
        raise ValueError("rows must contain at least one concept.")
    if top_k <= 0:
        raise ValueError("top_k must be positive.")
    if image.dim() == 3:
        image = image.unsqueeze(0)
    if image.dim() != 4 or image.size(0) != 1 or image.size(1) != 3:
        raise ValueError("image must have shape [3, H, W] or [1, 3, H, W].")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(16.0, 11.0))

    predicted_class = str(summary["cbm_predicted_class"])
    contrast_class = str(summary.get("contrast_class", predicted_class))
    fixed_contrast = contrast_class != predicted_class
    class_line = f"true: {summary['true_class']} | CBM top-1: {predicted_class}"
    if fixed_contrast:
        class_line += f"\nfixed contrast: {contrast_class} vs {summary['true_class']}"
    contrast_probability = float(
        summary.get("cbm_contrast_probability", summary["cbm_predicted_probability"])
    )

    axes[0, 0].imshow(_as_numpy_image(image, 0))
    axes[0, 0].set_title(
        "A. Original image\n"
        f"{class_line}"
    )
    axes[0, 0].axis("off")
    axes[0, 0].text(
        0.02,
        0.02,
        (
            f"confidence={float(summary['cbm_confidence']):.3f}\n"
            f"p(true)={float(summary['cbm_true_probability']):.3f}\n"
            f"p({contrast_class})={contrast_probability:.3f}\n"
            f"margin {contrast_class}-true={float(summary['wrong_vs_true_margin']):+.3f}"
        ),
        transform=axes[0, 0].transAxes,
        va="bottom",
        ha="left",
        fontsize=9.5,
        family="monospace",
        bbox={
            "boxstyle": "round,pad=0.45",
            "facecolor": "white",
            "edgecolor": "#cbd5e1",
            "alpha": 0.9,
        },
    )

    error_rows = sorted(
        rows,
        key=lambda row: float(row["distance_to_true_prototype"]),
        reverse=True,
    )[:top_k]
    error_rows.reverse()
    positions = np.arange(len(error_rows))
    axes[0, 1].barh(
        positions - 0.18,
        [float(row["predicted_value"]) for row in error_rows],
        height=0.34,
        color="#c2410c",
        label="predicted concept",
    )
    axes[0, 1].barh(
        positions + 0.18,
        [float(row["true_prototype_value"]) for row in error_rows],
        height=0.34,
        color="#0f766e",
        label="AwA2 true-class target",
    )
    axes[0, 1].set_yticks(positions, [str(row["concept"]) for row in error_rows])
    axes[0, 1].set_xlim(0.0, 1.0)
    axes[0, 1].set_xlabel("concept value")
    axes[0, 1].set_title("B. Largest concept prediction errors")
    axes[0, 1].grid(axis="x", alpha=0.25)
    axes[0, 1].legend(loc="lower right", fontsize=8.5)

    contribution_rows = sorted(
        rows,
        key=lambda row: abs(float(row["wrong_vs_true_margin_contribution"])),
        reverse=True,
    )[:top_k]
    contribution_rows.reverse()
    contribution_values = [
        float(row["wrong_vs_true_margin_contribution"])
        for row in contribution_rows
    ]
    axes[1, 0].barh(
        [str(row["concept"]) for row in contribution_rows],
        contribution_values,
        color=["#b91c1c" if value > 0 else "#0f766e" for value in contribution_values],
    )
    axes[1, 0].axvline(0.0, color="black", linewidth=0.9)
    axes[1, 0].set_xlabel(f"contribution to {contrast_class} - true logit margin")
    axes[1, 0].set_title(
        f"C. Linear class-head contributions ({contrast_class} vs {summary['true_class']})\n"
        f"bias contribution={float(summary['wrong_vs_true_bias']):+.3f}"
    )
    axes[1, 0].grid(axis="x", alpha=0.25)
    axes[1, 0].text(
        0.01,
        -0.17,
        f"red: supports {contrast_class} | green: supports {summary['true_class']}",
        transform=axes[1, 0].transAxes,
        fontsize=8.5,
    )

    ranked_intervention_rows = sorted(
        rows,
        key=lambda row: abs(float(row["correction_true_probability_delta"])),
        reverse=True,
    )
    intervention_rows = ranked_intervention_rows[:top_k]
    for recovery_row in (
        row
        for row in ranked_intervention_rows
        if bool(row.get("correction_recovers_true_class", False))
    ):
        if recovery_row in intervention_rows:
            continue
        replace_index = next(
            (
                index
                for index in range(len(intervention_rows) - 1, -1, -1)
                if not bool(
                    intervention_rows[index].get(
                        "correction_recovers_true_class",
                        False,
                    )
                )
            ),
            None,
        )
        if replace_index is not None:
            intervention_rows[replace_index] = recovery_row
    intervention_rows.reverse()
    intervention_values = [
        float(row["correction_true_probability_delta"])
        for row in intervention_rows
    ]
    axes[1, 1].barh(
        [
            f"{row['concept']}*"
            if bool(row.get("correction_recovers_true_class", False))
            else str(row["concept"])
            for row in intervention_rows
        ],
        intervention_values,
        color=[
            "#0f766e"
            if bool(row.get("correction_recovers_true_class", False))
            else "#5aa79d"
            if value > 0
            else "#b45309"
            for row, value in zip(intervention_rows, intervention_values, strict=True)
        ],
    )
    axes[1, 1].axvline(0.0, color="black", linewidth=0.9)
    axes[1, 1].set_xlabel("change in true-class probability after correction")
    axes[1, 1].set_title("D. One-concept oracle interventions")
    axes[1, 1].grid(axis="x", alpha=0.25)
    if any(
        bool(row.get("correction_recovers_true_class", False))
        for row in intervention_rows
    ):
        axes[1, 1].text(
            0.01,
            -0.17,
            "* correction recovers the true class",
            transform=axes[1, 1].transAxes,
            fontsize=8.5,
        )

    figure_title = (
        f"CBM Concept Decomposition: fixed {contrast_class} vs {summary['true_class']} contrast"
        if fixed_contrast
        else "CBM Error Decomposition: concept prediction, weighting, and intervention"
    )
    fig.suptitle(
        figure_title,
        fontsize=15,
        y=1.01,
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)
