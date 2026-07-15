"""Robustness and sanity-check utilities for saliency explanations."""

from __future__ import annotations

import copy
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import NoiseTunnel, Occlusion, Saliency
from torch import nn

from src.data import denormalize_batch
from src.xai import attributions_to_saliency_map, blurred_baseline, overlay_heatmap


def smoothgrad_saliency(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_samples: int = 12,
    noise_std: float = 0.12,
) -> torch.Tensor:
    """Compute SmoothGrad through Captum NoiseTunnel."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    gradient_inputs = inputs.detach().clone().requires_grad_(True)
    attributions = NoiseTunnel(Saliency(model)).attribute(
        gradient_inputs,
        nt_type="smoothgrad",
        nt_samples=n_samples,
        stdevs=noise_std,
        target=targets,
        abs=True,
    )
    return attributions_to_saliency_map(attributions).detach()


def occlusion_sensitivity(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    patch_size: int = 32,
    stride: int = 16,
    batch_size: int = 32,
) -> torch.Tensor:
    """Compute occlusion sensitivity through Captum Occlusion."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")

    model.eval()
    baselines = blurred_baseline(inputs)
    attributions = Occlusion(model).attribute(
        inputs,
        sliding_window_shapes=(inputs.size(1), patch_size, patch_size),
        strides=(inputs.size(1), stride, stride),
        baselines=baselines,
        target=targets,
        perturbations_per_eval=batch_size,
    )
    return attributions_to_saliency_map(attributions, positive_only=True).detach()


def randomized_copy(model: nn.Module, seed: int = 42) -> nn.Module:
    """Return a copy of the model with randomly reinitialized learnable modules."""
    torch.manual_seed(seed)
    copied = copy.deepcopy(model)
    for module in copied.modules():
        reset = getattr(module, "reset_parameters", None)
        if callable(reset):
            reset()
    return copied


def topk_iou(first: torch.Tensor, second: torch.Tensor, fraction: float = 0.2) -> float:
    """IoU between the top-k most salient pixels of two normalized maps."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1.")
    first_flat = first.detach().flatten()
    second_flat = second.detach().flatten()
    k = max(1, int(first_flat.numel() * fraction))
    first_indices = torch.topk(first_flat, k=k).indices
    second_indices = torch.topk(second_flat, k=k).indices
    first_mask = torch.zeros_like(first_flat, dtype=torch.bool)
    second_mask = torch.zeros_like(second_flat, dtype=torch.bool)
    first_mask[first_indices] = True
    second_mask[second_indices] = True
    intersection = torch.logical_and(first_mask, second_mask).sum().item()
    union = torch.logical_or(first_mask, second_mask).sum().item()
    return float(intersection / max(union, 1))


def rank_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Spearman-like rank correlation for flattened saliency tensors."""
    x = first.detach().flatten().float()
    y = second.detach().flatten().float()
    x_rank = torch.argsort(torch.argsort(x)).float()
    y_rank = torch.argsort(torch.argsort(y)).float()
    x_rank = x_rank - x_rank.mean()
    y_rank = y_rank - y_rank.mean()
    denominator = x_rank.norm() * y_rank.norm()
    if denominator <= 1e-12:
        return 0.0
    return float(torch.dot(x_rank, y_rank).item() / denominator.item())


def saliency_pair_metrics(
    reference: torch.Tensor,
    candidate: torch.Tensor,
    prefix: str,
    top_fraction: float = 0.2,
) -> dict[str, float]:
    """Compute overlap and rank-stability metrics for two saliency batches."""
    rows = {}
    ious = []
    correlations = []
    for index in range(reference.size(0)):
        ious.append(topk_iou(reference[index], candidate[index], fraction=top_fraction))
        correlations.append(rank_correlation(reference[index], candidate[index]))
    rows[f"{prefix}_iou_top20_mean"] = float(np.mean(ious))
    rows[f"{prefix}_spearman_mean"] = float(np.mean(correlations))
    return rows


def save_audit_grid(
    images: torch.Tensor,
    vanilla_maps: torch.Tensor,
    smoothgrad_maps: torch.Tensor,
    occlusion_maps: torch.Tensor,
    randomized_maps: torch.Tensor,
    true_names: list[str],
    predicted_names: list[str],
    confidences: list[float],
    output_path: str | Path,
) -> None:
    """Save visual comparison for Phase 9 explanation audits."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    denorm = denormalize_batch(images.detach().cpu()).clamp(0, 1)
    n_images = images.size(0)

    fig, axes = plt.subplots(n_images, 5, figsize=(16.5, 3.2 * n_images))
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for index in range(n_images):
        image_np = denorm[index].permute(1, 2, 0).numpy()
        maps = [
            vanilla_maps[index, 0].detach().cpu().numpy(),
            smoothgrad_maps[index, 0].detach().cpu().numpy(),
            occlusion_maps[index, 0].detach().cpu().numpy(),
            randomized_maps[index, 0].detach().cpu().numpy(),
        ]
        titles = [
            "Input gradients",
            "SmoothGrad",
            "Occlusion sensitivity",
            "Randomized-model gradients",
        ]

        axes[index, 0].imshow(image_np)
        axes[index, 0].set_title(
            f"image\ntrue={true_names[index]}\npred={predicted_names[index]} ({confidences[index]:.2f})",
            fontsize=9,
        )
        axes[index, 0].axis("off")

        for col, (saliency, title) in enumerate(zip(maps, titles), start=1):
            axes[index, col].imshow(overlay_heatmap(image_np, saliency))
            axes[index, col].set_title(title, fontsize=9)
            axes[index, col].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
