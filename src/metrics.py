"""Quantitative metrics for saliency-map stress tests."""

from __future__ import annotations

import math

import torch


def _average_tie_ranks(values: torch.Tensor) -> torch.Tensor:
    """Return zero-based average ranks for each row of a rank-2 tensor.

    Equal values receive the mean of the ranks that they occupy. This is the
    tie convention used by Spearman's rank correlation.
    """
    if values.dim() != 2:
        raise ValueError("values must have shape [batch, features].")

    rank_dtype = torch.float64 if values.dtype == torch.float64 else torch.float32
    ranks = torch.empty(values.shape, dtype=rank_dtype, device=values.device)

    for row_index in range(values.size(0)):
        sorted_values, sorted_indices = torch.sort(values[row_index].detach())
        _unique_values, counts = torch.unique_consecutive(
            sorted_values,
            return_counts=True,
        )
        group_ends = torch.cumsum(counts, dim=0)
        group_starts = group_ends - counts
        average_group_ranks = (
            (group_starts + group_ends - 1).to(dtype=rank_dtype) / 2.0
        )
        sorted_ranks = torch.repeat_interleave(average_group_ranks, counts)
        ranks[row_index].scatter_(0, sorted_indices, sorted_ranks)

    return ranks


def percentage_token(percent: float) -> str:
    """Return a stable CSV-safe token for a percentage value."""
    if not math.isfinite(percent):
        raise ValueError("percent must be finite.")
    rounded = round(float(percent), 10)
    text = f"{rounded:.10f}".rstrip("0").rstrip(".")
    return text.replace(".", "p")


def top_fraction_label(fraction: float) -> str:
    """Return labels such as ``top20`` or ``top12p5`` for CSV columns."""
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")
    return f"top{percentage_token(fraction * 100.0)}"


def top_fraction_mask(maps: torch.Tensor, fraction: float) -> torch.Tensor:
    """Select an exact, deterministic fraction of each flattened map.

    For a positive fraction, ``max(1, floor(N * fraction))`` values are
    selected. Ties are resolved by their original flattened position through a
    stable descending sort. This keeps deletion/insertion masks nested and
    prevents threshold ties from selecting more than the requested count.
    """
    if maps.dim() < 2:
        raise ValueError("maps must include a batch dimension.")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("fraction must be in [0, 1].")

    flat = maps.detach().flatten(start_dim=1)
    if flat.size(1) == 0:
        raise ValueError("maps must not be empty.")
    if not torch.isfinite(flat).all():
        raise ValueError("maps must contain only finite values.")

    mask = torch.zeros_like(flat, dtype=torch.bool)
    if fraction == 0.0:
        return mask.reshape_as(maps)

    count = min(flat.size(1), max(1, int(flat.size(1) * fraction)))
    ranked_indices = torch.argsort(
        flat,
        dim=1,
        descending=True,
        stable=True,
    )
    mask.scatter_(1, ranked_indices[:, :count], True)
    return mask.reshape_as(maps)


def saliency_iou_at_top_percent(
    original_maps: torch.Tensor,
    perturbed_maps: torch.Tensor,
    top_percent: float = 20.0,
) -> torch.Tensor:
    """Return IoU between binary masks of the top salient pixels."""
    if not 0.0 < top_percent < 100.0:
        raise ValueError("top_percent must be in (0, 100).")
    if original_maps.shape != perturbed_maps.shape:
        raise ValueError("saliency map batches must have the same shape.")

    fraction = top_percent / 100.0
    original_mask = top_fraction_mask(original_maps, fraction)
    perturbed_mask = top_fraction_mask(perturbed_maps, fraction)
    intersection = (original_mask & perturbed_mask).flatten(start_dim=1).sum(dim=1).float()
    union = (original_mask | perturbed_mask).flatten(start_dim=1).sum(dim=1).float()
    return intersection / union.clamp_min(1.0)


def spearman_rank_correlation(
    original_maps: torch.Tensor,
    perturbed_maps: torch.Tensor,
) -> torch.Tensor:
    """Return tie-aware Spearman correlation for flattened saliency maps.

    The result is ``NaN`` for a sample when either map is constant or contains
    non-finite values, because rank correlation is undefined in those cases.
    """
    if original_maps.dim() < 2 or perturbed_maps.dim() < 2:
        raise ValueError("saliency maps must include a batch dimension.")
    if original_maps.size(0) != perturbed_maps.size(0):
        raise ValueError("saliency map batches must have the same size.")

    original_flat = original_maps.flatten(start_dim=1)
    perturbed_flat = perturbed_maps.flatten(start_dim=1)
    if original_flat.size(1) != perturbed_flat.size(1):
        raise ValueError("saliency maps must contain the same number of values per sample.")
    if original_flat.size(1) == 0:
        raise ValueError("saliency maps must not be empty.")

    valid_rows = torch.isfinite(original_flat).all(dim=1) & torch.isfinite(
        perturbed_flat
    ).all(dim=1)
    original_ranks = _average_tie_ranks(original_flat)
    perturbed_ranks = _average_tie_ranks(perturbed_flat)
    original_ranks = original_ranks - original_ranks.mean(dim=1, keepdim=True)
    perturbed_ranks = perturbed_ranks - perturbed_ranks.mean(dim=1, keepdim=True)

    numerator = (original_ranks * perturbed_ranks).sum(dim=1)
    denominator = original_ranks.norm(dim=1) * perturbed_ranks.norm(dim=1)
    defined = valid_rows & (denominator > torch.finfo(denominator.dtype).eps)
    correlation = torch.full_like(numerator, float("nan"))
    correlation[defined] = numerator[defined] / denominator[defined]
    return correlation.clamp(min=-1.0, max=1.0)
