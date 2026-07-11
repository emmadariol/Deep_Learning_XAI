"""Advanced attribution audits for image-classification explanations.

The utilities in this module treat saliency maps as hypotheses that must be
tested quantitatively. They complement visual Grad-CAM/Integrated Gradients
inspection with faithfulness, stability, region-allocation and
class-discriminativeness diagnostics.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import IMAGENET_MEAN, IMAGENET_STD, denormalize_batch
from src.explainability_audit import rank_correlation, topk_iou
from src.perturb import make_background_mask
from src.xai import (
    GradCAM,
    ScoreCAM,
    blurred_baseline,
    expected_gradients_maps,
    input_gradient_saliency,
    integrated_gradients_maps,
    normalize_maps,
    overlay_heatmap,
)


@dataclass(frozen=True)
class AttributionBundle:
    """Attribution tensors and optional method-specific internals."""

    maps: torch.Tensor
    raw_attributions: torch.Tensor | None = None
    baseline: torch.Tensor | None = None


def predict_with_logits(model: nn.Module, inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return logits, predicted labels and predicted confidences."""
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
    return logits, predictions, confidences


def target_probabilities(model: nn.Module, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return probabilities assigned to explicit target classes."""
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(inputs), dim=1)
    return probabilities.gather(1, targets.view(-1, 1)).squeeze(1)


def compute_attribution(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    ig_steps: int = 24,
    ig_internal_batch_size: int | None = 4,
    blur_radius: float = 18.0,
    scorecam_max_channels: int | None = 64,
    scorecam_batch_size: int = 16,
    expected_gradients_samples: int = 24,
    expected_gradients_baselines: torch.Tensor | None = None,
) -> AttributionBundle:
    """Compute one normalized attribution map batch.

    Supported methods are ``gradcam``, ``scorecam``, ``integrated_gradients``,
    ``expected_gradients`` and ``input_gradients``. Maps are always returned
    with shape ``[B, 1, H, W]``.
    """
    if method == "gradcam":
        gradcam = GradCAM(model, model.layer4[-1])
        try:
            return AttributionBundle(maps=gradcam(inputs, targets))
        finally:
            gradcam.close()
    if method == "scorecam":
        scorecam = ScoreCAM(
            model,
            model.layer4[-1],
            max_channels=scorecam_max_channels,
            batch_size=scorecam_batch_size,
            blur_radius=blur_radius,
        )
        try:
            return AttributionBundle(maps=scorecam(inputs, targets))
        finally:
            scorecam.close()
    if method == "integrated_gradients":
        maps, raw_attributions, baseline = integrated_gradients_maps(
            model=model,
            inputs=inputs,
            targets=targets,
            steps=ig_steps,
            internal_batch_size=ig_internal_batch_size,
            blur_radius=blur_radius,
        )
        return AttributionBundle(maps=maps, raw_attributions=raw_attributions, baseline=baseline)
    if method == "expected_gradients":
        maps, raw_attributions, baseline_pool = expected_gradients_maps(
            model=model,
            inputs=inputs,
            targets=targets,
            baselines=expected_gradients_baselines,
            n_samples=expected_gradients_samples,
            internal_batch_size=ig_internal_batch_size,
            blur_radius=blur_radius,
        )
        return AttributionBundle(maps=maps, raw_attributions=raw_attributions, baseline=baseline_pool)
    if method == "input_gradients":
        return AttributionBundle(maps=input_gradient_saliency(model, inputs, targets))
    raise ValueError(
        "method must be one of: gradcam, scorecam, integrated_gradients, "
        "expected_gradients, input_gradients"
    )


def saliency_entropy(maps: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Return normalized spatial entropy for each saliency map.

    The entropy is divided by ``log(number_of_pixels)`` so values are in
    approximately ``[0, 1]``. High values indicate diffuse saliency.
    """
    flat = maps.flatten(start_dim=1).float()
    probabilities = flat / flat.sum(dim=1, keepdim=True).clamp_min(eps)
    entropy = -(probabilities * (probabilities + eps).log()).sum(dim=1)
    return entropy / np.log(flat.size(1))


def region_saliency_scores(
    maps: torch.Tensor,
    images: torch.Tensor,
    mask_strategy: str = "center_ellipse",
    foreground_scale: float = 0.68,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Measure how much saliency lies on approximate animal and background regions.

    AwA2 does not provide segmentation masks. The foreground is therefore an
    explicit approximation: the central ellipse or box is treated as animal
    proxy, while the complement is treated as background.
    """
    background_mask = make_background_mask(
        images,
        strategy=mask_strategy,
        foreground_scale=foreground_scale,
    ).to(maps.device)
    animal_mask = ~background_mask
    total = maps.flatten(start_dim=1).sum(dim=1).clamp_min(eps)
    animal_saliency = (maps * animal_mask.float()).flatten(start_dim=1).sum(dim=1) / total
    background_saliency = (maps * background_mask.float()).flatten(start_dim=1).sum(dim=1) / total
    return animal_saliency, background_saliency, background_mask


def _top_fraction_mask(maps: torch.Tensor, fraction: float) -> torch.Tensor:
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")
    if fraction == 0.0:
        return torch.zeros_like(maps, dtype=torch.bool)
    flat = maps.flatten(start_dim=1)
    k = max(1, int(flat.size(1) * fraction))
    threshold = torch.topk(flat, k=k, dim=1).values[:, -1].view(-1, 1, 1, 1)
    return maps >= threshold


def _replace_by_mask(original: torch.Tensor, replacement: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    mask_rgb = mask.expand_as(original)
    return torch.where(mask_rgb, replacement, original)


def deletion_insertion_curves(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    maps: torch.Tensor,
    fractions: list[float],
    baseline: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute deletion and insertion target-probability curves.

    Deletion starts from the original image and replaces the most salient pixels
    with the baseline. Insertion starts from the baseline and restores the most
    salient pixels from the original image.
    """
    if baseline is None:
        baseline = blurred_baseline(inputs)

    deletion_scores: list[torch.Tensor] = []
    insertion_scores: list[torch.Tensor] = []
    for fraction in fractions:
        salient_mask = _top_fraction_mask(maps, fraction)
        deleted = _replace_by_mask(inputs, baseline, salient_mask)
        inserted = _replace_by_mask(baseline, inputs, salient_mask)
        deletion_scores.append(target_probabilities(model, deleted, targets))
        insertion_scores.append(target_probabilities(model, inserted, targets))
    return torch.stack(deletion_scores, dim=1), torch.stack(insertion_scores, dim=1)


def trapezoid_auc(x_values: list[float], y_values: torch.Tensor) -> torch.Tensor:
    """Return per-example trapezoidal area under a curve."""
    x = torch.tensor(x_values, device=y_values.device, dtype=y_values.dtype)
    widths = x[1:] - x[:-1]
    heights = 0.5 * (y_values[:, 1:] + y_values[:, :-1])
    return (widths.view(1, -1) * heights).sum(dim=1)


def sensitivity_to_noise(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    method: str,
    noise_std: float = 0.03,
    ig_steps: int = 16,
    ig_internal_batch_size: int | None = 4,
    blur_radius: float = 18.0,
    scorecam_max_channels: int | None = 64,
    scorecam_batch_size: int = 16,
    expected_gradients_samples: int = 12,
    expected_gradients_baselines: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare attribution maps before and after a small input perturbation."""
    reference = compute_attribution(
        model,
        inputs,
        targets,
        method=method,
        ig_steps=ig_steps,
        ig_internal_batch_size=ig_internal_batch_size,
        blur_radius=blur_radius,
        scorecam_max_channels=scorecam_max_channels,
        scorecam_batch_size=scorecam_batch_size,
        expected_gradients_samples=expected_gradients_samples,
        expected_gradients_baselines=expected_gradients_baselines,
    ).maps
    noisy_inputs = (inputs + torch.randn_like(inputs) * noise_std).detach()
    candidate = compute_attribution(
        model,
        noisy_inputs,
        targets,
        method=method,
        ig_steps=ig_steps,
        ig_internal_batch_size=ig_internal_batch_size,
        blur_radius=blur_radius,
        scorecam_max_channels=scorecam_max_channels,
        scorecam_batch_size=scorecam_batch_size,
        expected_gradients_samples=expected_gradients_samples,
        expected_gradients_baselines=expected_gradients_baselines,
    ).maps
    with torch.no_grad():
        reference_predictions = torch.softmax(model(inputs), dim=1).argmax(dim=1)
        noisy_predictions = torch.softmax(model(noisy_inputs), dim=1).argmax(dim=1)
    same_prediction = reference_predictions == noisy_predictions
    return reference, candidate, same_prediction


def class_discriminativeness(
    model: nn.Module,
    inputs: torch.Tensor,
    method: str,
    ig_steps: int = 16,
    ig_internal_batch_size: int | None = 4,
    blur_radius: float = 18.0,
    top_fraction: float = 0.2,
    scorecam_max_channels: int | None = 64,
    scorecam_batch_size: int = 16,
    expected_gradients_samples: int = 12,
    expected_gradients_baselines: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare maps for top-1 and top-2 predicted classes on the same image."""
    logits, top1, _confidence = predict_with_logits(model, inputs)
    top2 = torch.topk(logits, k=min(2, logits.size(1)), dim=1).indices[:, -1]
    top1_maps = compute_attribution(
        model,
        inputs,
        top1,
        method=method,
        ig_steps=ig_steps,
        ig_internal_batch_size=ig_internal_batch_size,
        blur_radius=blur_radius,
        scorecam_max_channels=scorecam_max_channels,
        scorecam_batch_size=scorecam_batch_size,
        expected_gradients_samples=expected_gradients_samples,
        expected_gradients_baselines=expected_gradients_baselines,
    ).maps
    top2_maps = compute_attribution(
        model,
        inputs,
        top2,
        method=method,
        ig_steps=ig_steps,
        ig_internal_batch_size=ig_internal_batch_size,
        blur_radius=blur_radius,
        scorecam_max_channels=scorecam_max_channels,
        scorecam_batch_size=scorecam_batch_size,
        expected_gradients_samples=expected_gradients_samples,
        expected_gradients_baselines=expected_gradients_baselines,
    ).maps
    ious = torch.tensor(
        [topk_iou(top1_maps[index], top2_maps[index], top_fraction) for index in range(inputs.size(0))],
        device=inputs.device,
    )
    correlations = torch.tensor(
        [rank_correlation(top1_maps[index], top2_maps[index]) for index in range(inputs.size(0))],
        device=inputs.device,
    )
    return top1, top2, top1_maps, top2_maps, torch.stack((ious, correlations), dim=1)


def black_baseline_like(inputs: torch.Tensor) -> torch.Tensor:
    """Return a normalized black-image baseline with the same shape as inputs."""
    image_space_black = torch.zeros_like(denormalize_batch(inputs))
    mean = torch.tensor(IMAGENET_MEAN, device=inputs.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=inputs.device).view(1, 3, 1, 1)
    return (image_space_black - mean) / std


def integrated_gradients_baseline_comparison(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 16,
    internal_batch_size: int | None = 4,
    blur_radius: float = 18.0,
    top_fraction: float = 0.2,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compare IG maps from blurred and black baselines."""
    blurred_maps, _blurred_attr, _blurred = integrated_gradients_maps(
        model=model,
        inputs=inputs,
        targets=targets,
        steps=steps,
        internal_batch_size=internal_batch_size,
        blur_radius=blur_radius,
    )
    black_baseline = black_baseline_like(inputs)
    from captum.attr import IntegratedGradients

    attributions = IntegratedGradients(model).attribute(
        inputs,
        baselines=black_baseline,
        target=targets,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
    )
    black_maps = normalize_maps(attributions.abs().sum(dim=1, keepdim=True))
    metrics = torch.tensor(
        [
            [topk_iou(blurred_maps[index], black_maps[index], top_fraction), rank_correlation(blurred_maps[index], black_maps[index])]
            for index in range(inputs.size(0))
        ],
        device=inputs.device,
    )
    return blurred_maps, black_maps, metrics


def save_deletion_insertion_plot(
    fractions: list[float],
    deletion_scores: torch.Tensor,
    insertion_scores: torch.Tensor,
    method: str,
    output_path: str | Path,
) -> None:
    """Save aggregate deletion/insertion curves."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    deletion_mean = deletion_scores.detach().cpu().float().mean(dim=0).numpy()
    insertion_mean = insertion_scores.detach().cpu().float().mean(dim=0).numpy()

    fig, axis = plt.subplots(figsize=(7.0, 4.4))
    axis.plot(fractions, deletion_mean, marker="o", label="Deletion")
    axis.plot(fractions, insertion_mean, marker="o", label="Insertion")
    axis.set_xlabel("Fraction of most salient pixels modified")
    axis.set_ylabel("Target-class probability")
    axis.set_title(f"Faithfulness curves - {method}")
    axis.grid(alpha=0.25)
    axis.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def save_class_discriminativeness_grid(
    images: torch.Tensor,
    top1_maps: torch.Tensor,
    top2_maps: torch.Tensor,
    true_names: list[str],
    top1_names: list[str],
    top2_names: list[str],
    output_path: str | Path,
) -> None:
    """Save visual comparison between top-1 and top-2 target explanations."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    denorm = denormalize_batch(images.detach().cpu()).clamp(0, 1)
    row_count = images.size(0)
    fig, axes = plt.subplots(row_count, 3, figsize=(10.5, 3.2 * row_count))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for index in range(row_count):
        image_np = denorm[index].permute(1, 2, 0).numpy()
        axes[index, 0].imshow(image_np)
        axes[index, 0].set_title(f"image\ntrue={true_names[index]}")
        axes[index, 1].imshow(overlay_heatmap(image_np, top1_maps[index, 0].detach().cpu().numpy()))
        axes[index, 1].set_title(f"top-1 target\n{top1_names[index]}")
        axes[index, 2].imshow(overlay_heatmap(image_np, top2_maps[index, 0].detach().cpu().numpy()))
        axes[index, 2].set_title(f"top-2 target\n{top2_names[index]}")
        for axis in axes[index]:
            axis.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Write rows to CSV."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
