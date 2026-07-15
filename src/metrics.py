"""Quantitative metrics for saliency-map stress tests."""

from __future__ import annotations

import torch


def saliency_iou_at_top_percent(
    original_maps: torch.Tensor,
    perturbed_maps: torch.Tensor,
    top_percent: float = 20.0,
) -> torch.Tensor:
    """Return IoU between binary masks of the top salient pixels."""
    if not 0.0 < top_percent < 100.0:
        raise ValueError("top_percent must be in (0, 100).")

    quantile = 1.0 - (top_percent / 100.0)
    original_flat = original_maps.flatten(start_dim=1)
    perturbed_flat = perturbed_maps.flatten(start_dim=1)
    original_threshold = torch.quantile(original_flat, quantile, dim=1).view(-1, 1, 1, 1)
    perturbed_threshold = torch.quantile(perturbed_flat, quantile, dim=1).view(-1, 1, 1, 1)

    original_mask = original_maps >= original_threshold
    perturbed_mask = perturbed_maps >= perturbed_threshold
    intersection = (original_mask & perturbed_mask).flatten(start_dim=1).sum(dim=1).float()
    union = (original_mask | perturbed_mask).flatten(start_dim=1).sum(dim=1).float()
    return intersection / union.clamp_min(1.0)


def spearman_rank_correlation(
    original_maps: torch.Tensor,
    perturbed_maps: torch.Tensor,
) -> torch.Tensor:
    """Return Spearman rank correlation for flattened saliency maps."""
    original_flat = original_maps.flatten(start_dim=1)
    perturbed_flat = perturbed_maps.flatten(start_dim=1)

    original_ranks = torch.argsort(torch.argsort(original_flat, dim=1), dim=1).float()
    perturbed_ranks = torch.argsort(torch.argsort(perturbed_flat, dim=1), dim=1).float()
    original_ranks = original_ranks - original_ranks.mean(dim=1, keepdim=True)
    perturbed_ranks = perturbed_ranks - perturbed_ranks.mean(dim=1, keepdim=True)

    numerator = (original_ranks * perturbed_ranks).sum(dim=1)
    denominator = original_ranks.norm(dim=1) * perturbed_ranks.norm(dim=1)
    return numerator / denominator.clamp_min(1e-8)
