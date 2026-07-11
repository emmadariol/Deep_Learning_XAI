"""Robustness and sanity-check utilities for saliency explanations."""

from __future__ import annotations

import copy
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from src.data import denormalize_batch
from src.xai import blurred_baseline, input_gradient_saliency, normalize_maps, overlay_heatmap


def smoothgrad_saliency(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    n_samples: int = 12,
    noise_std: float = 0.12,
) -> torch.Tensor:
    """Average input-gradient saliency over noisy copies of the same image."""
    if n_samples <= 0:
        raise ValueError("n_samples must be positive.")

    maps: list[torch.Tensor] = []
    for _index in range(n_samples):
        noise = torch.randn_like(inputs) * noise_std
        noisy_inputs = (inputs + noise).detach()
        maps.append(input_gradient_saliency(model, noisy_inputs, targets))
    return normalize_maps(torch.stack(maps, dim=0).mean(dim=0))


def occlusion_sensitivity(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    patch_size: int = 32,
    stride: int = 16,
    batch_size: int = 32,
) -> torch.Tensor:
    """Measure target-probability drop after replacing local patches with blur."""
    if patch_size <= 0 or stride <= 0:
        raise ValueError("patch_size and stride must be positive.")

    model.eval()
    device = inputs.device
    baselines = blurred_baseline(inputs)
    maps = torch.zeros((inputs.size(0), 1, inputs.size(2), inputs.size(3)), device=device)
    counts = torch.zeros_like(maps)

    with torch.no_grad():
        original_probs = torch.softmax(model(inputs), dim=1).gather(1, targets.view(-1, 1))

    for sample_index in range(inputs.size(0)):
        occluded_images: list[torch.Tensor] = []
        windows: list[tuple[int, int, int, int]] = []
        for top in range(0, inputs.size(2), stride):
            bottom = min(top + patch_size, inputs.size(2))
            top = max(0, bottom - patch_size)
            for left in range(0, inputs.size(3), stride):
                right = min(left + patch_size, inputs.size(3))
                left = max(0, right - patch_size)
                occluded = inputs[sample_index].clone()
                occluded[:, top:bottom, left:right] = baselines[sample_index, :, top:bottom, left:right]
                occluded_images.append(occluded)
                windows.append((top, bottom, left, right))

        drops: list[torch.Tensor] = []
        for start in range(0, len(occluded_images), batch_size):
            batch = torch.stack(occluded_images[start : start + batch_size], dim=0)
            with torch.no_grad():
                probs = torch.softmax(model(batch), dim=1)[:, targets[sample_index]]
            drops.append(original_probs[sample_index, 0] - probs)
        all_drops = torch.cat(drops, dim=0)

        for drop, (top, bottom, left, right) in zip(all_drops, windows):
            maps[sample_index, 0, top:bottom, left:right] += torch.clamp(drop, min=0.0)
            counts[sample_index, 0, top:bottom, left:right] += 1.0

    maps = maps / counts.clamp_min(1.0)
    return normalize_maps(maps)


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


def write_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Write rows to CSV using the keys of the first row as fieldnames."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"No rows to write for {output_path}")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
