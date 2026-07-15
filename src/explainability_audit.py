"""Generic saliency-map comparison utilities.

This module intentionally contains only method-agnostic metrics. Attribution
methods themselves live in ``src.xai`` and are limited to the maintained project
set: Grad-CAM and Integrated Gradients.
"""

from __future__ import annotations

import numpy as np
import torch

from src.metrics import spearman_rank_correlation, top_fraction_label, top_fraction_mask


def topk_iou(first: torch.Tensor, second: torch.Tensor, fraction: float = 0.2) -> float:
    """IoU between the top-k most salient pixels of two normalized maps."""
    if not 0.0 < fraction < 1.0:
        raise ValueError("fraction must be between 0 and 1.")
    first_flat = first.detach().flatten()
    second_flat = second.detach().flatten()
    if first_flat.numel() != second_flat.numel():
        raise ValueError("saliency maps must contain the same number of values.")
    first_mask = top_fraction_mask(first_flat.reshape(1, -1), fraction).flatten()
    second_mask = top_fraction_mask(second_flat.reshape(1, -1), fraction).flatten()
    intersection = torch.logical_and(first_mask, second_mask).sum().item()
    union = torch.logical_or(first_mask, second_mask).sum().item()
    return float(intersection / max(union, 1))


def rank_correlation(first: torch.Tensor, second: torch.Tensor) -> float:
    """Tie-aware Spearman correlation for two saliency tensors."""
    correlation = spearman_rank_correlation(
        first.detach().reshape(1, -1),
        second.detach().reshape(1, -1),
    )
    return float(correlation[0].item())


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
    finite_correlations = [value for value in correlations if np.isfinite(value)]
    top_label = top_fraction_label(top_fraction)
    rows[f"{prefix}_iou_{top_label}_mean"] = float(np.mean(ious))
    rows[f"{prefix}_spearman_mean"] = (
        float(np.mean(finite_correlations)) if finite_correlations else float("nan")
    )
    rows[f"{prefix}_spearman_valid_count"] = len(finite_correlations)
    rows[f"{prefix}_spearman_undefined_count"] = len(correlations) - len(finite_correlations)
    return rows
