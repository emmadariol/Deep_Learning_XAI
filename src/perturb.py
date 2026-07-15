"""Background perturbations for AwA2 stress tests."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from src.data import IMAGENET_MEAN, IMAGENET_STD, denormalize_batch

LOGGER = logging.getLogger(__name__)


def normalize_batch(images: torch.Tensor) -> torch.Tensor:
    """Apply ImageNet normalization to images in [0, 1]."""
    mean = torch.tensor(IMAGENET_MEAN, device=images.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=images.device).view(1, 3, 1, 1)
    return (images - mean) / std


def make_background_mask(
    images: torch.Tensor,
    strategy: str = "center_ellipse",
    foreground_scale: float = 0.68,
) -> torch.Tensor:
    """Return a boolean mask for pixels treated as background.

    AwA2 does not provide segmentation masks. This function therefore provides
    explicit, reproducible approximations:

    - ``center_ellipse`` keeps a centered elliptical region clean.
    - ``center_box`` keeps a centered rectangular region clean.
    - ``global`` treats the whole image as background, useful as a fallback.
    """
    if images.dim() != 4:
        raise ValueError("images must have shape [B, C, H, W].")
    if not math.isfinite(foreground_scale) or not 0.0 < foreground_scale < 1.0:
        raise ValueError("foreground_scale must be in (0, 1).")

    batch_size, _channels, height, width = images.shape
    device = images.device

    if strategy == "global":
        return torch.ones((batch_size, 1, height, width), dtype=torch.bool, device=device)

    yy = torch.linspace(-1.0, 1.0, height, device=device).view(1, 1, height, 1)
    xx = torch.linspace(-1.0, 1.0, width, device=device).view(1, 1, 1, width)

    if strategy == "center_ellipse":
        foreground = (xx / foreground_scale).pow(2) + (yy / foreground_scale).pow(2) <= 1.0
    elif strategy == "center_box":
        foreground = (xx.abs() <= foreground_scale) & (yy.abs() <= foreground_scale)
    else:
        raise ValueError(
            "Unsupported mask strategy. Use one of: center_ellipse, center_box, global."
        )

    background = ~foreground
    return background.expand(batch_size, 1, height, width)


def perturb_background(
    inputs: torch.Tensor,
    background_mask: torch.Tensor,
    method: str,
    noise_std: float = 0.25,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Apply one perturbation method to the selected background pixels."""
    if inputs.dim() != 4:
        raise ValueError("inputs must have shape [B, C, H, W].")
    if not math.isfinite(noise_std) or noise_std < 0.0:
        raise ValueError("noise_std must be a non-negative finite number.")
    if background_mask.shape != (inputs.size(0), 1, inputs.size(2), inputs.size(3)):
        raise ValueError("background_mask must have shape [B, 1, H, W] matching inputs.")
    images = denormalize_batch(inputs.detach()).clamp(0.0, 1.0)
    mask = background_mask.to(device=images.device, dtype=torch.bool)
    mask_rgb = mask.expand_as(images)

    if method == "gaussian_noise":
        noise = torch.randn(
            images.shape,
            generator=generator,
            device=images.device,
            dtype=images.dtype,
        ) * noise_std
        replacement = (images + noise).clamp(0.0, 1.0)
    elif method == "color_shift":
        replacement = 1.0 - images
    elif method == "background_swap":
        replacement = torch.rand(
            images.shape,
            generator=generator,
            device=images.device,
            dtype=images.dtype,
        )
    else:
        raise ValueError(
            "Unsupported perturbation method. Use gaussian_noise, color_shift, or background_swap."
        )

    perturbed = torch.where(mask_rgb, replacement, images)
    normalized = normalize_batch(perturbed)
    log_tensor_stats(f"perturb.{method}.image_space", perturbed)
    log_tensor_stats(f"perturb.{method}.normalized", normalized)
    return normalized


def apply_perturbation_suite(
    inputs: torch.Tensor,
    mask_strategy: str = "center_ellipse",
    foreground_scale: float = 0.68,
    methods: tuple[str, ...] = ("gaussian_noise", "color_shift", "background_swap"),
    noise_std: float = 0.25,
    seed: int = 42,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Create the background mask and all requested perturbed batches."""
    if not methods:
        raise ValueError("methods must contain at least one perturbation.")
    background_mask = make_background_mask(
        inputs,
        strategy=mask_strategy,
        foreground_scale=foreground_scale,
    )
    generator = torch.Generator(device=inputs.device)
    generator.manual_seed(seed)

    perturbed = {
        method: perturb_background(
            inputs=inputs,
            background_mask=background_mask,
            method=method,
            noise_std=noise_std,
            generator=generator,
        )
        for method in methods
    }
    log_tensor_stats("perturb.background_mask", background_mask.float())
    return background_mask, perturbed


def predict_batch(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return predicted labels and confidences."""
    predictions, confidences, _probabilities = predict_batch_probabilities(
        model,
        images,
    )
    return predictions, confidences


def predict_batch_probabilities(
    model: torch.nn.Module,
    images: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return predicted labels, top-1 confidences and all class probabilities."""
    model.eval()
    with torch.no_grad():
        logits = model(images)
        probabilities = torch.softmax(logits, dim=1)
        confidences, predictions = probabilities.max(dim=1)
    return predictions.detach(), confidences.detach(), probabilities.detach()


def probabilities_for_targets(
    probabilities: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Select each sample's probability for a fixed target class."""
    if probabilities.dim() != 2:
        raise ValueError("probabilities must have shape [B, C].")
    if targets.dim() != 1 or targets.size(0) != probabilities.size(0):
        raise ValueError("targets must have shape [B] matching probabilities.")
    if targets.dtype == torch.bool or targets.dtype.is_floating_point:
        raise ValueError("targets must contain integer class indices.")
    targets = targets.to(device=probabilities.device, dtype=torch.long)
    if targets.numel() and (
        (targets < 0).any() or (targets >= probabilities.size(1)).any()
    ):
        raise ValueError("targets contain a class index outside the probability matrix.")
    return probabilities.gather(1, targets.unsqueeze(1)).squeeze(1)


def save_perturbation_grid(
    original_images: torch.Tensor,
    background_mask: torch.Tensor,
    perturbed_batches: dict[str, torch.Tensor],
    true_names: list[str],
    original_pred_names: list[str],
    perturbed_pred_names: dict[str, list[str]],
    output_path: str | Path,
) -> None:
    """Save a visual grid with original image, mask and perturbation variants."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    original_denorm = denormalize_batch(original_images.detach().cpu()).clamp(0.0, 1.0)
    perturbed_denorm = {
        name: denormalize_batch(batch.detach().cpu()).clamp(0.0, 1.0)
        for name, batch in perturbed_batches.items()
    }
    mask_cpu = background_mask.detach().cpu().float()
    row_count = original_images.size(0)
    method_names = list(perturbed_batches)
    col_count = 2 + len(method_names)

    fig, axes = plt.subplots(row_count, col_count, figsize=(3.2 * col_count, 3.0 * row_count))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    for row in range(row_count):
        original_np = original_denorm[row].permute(1, 2, 0).numpy()
        axes[row, 0].imshow(original_np)
        axes[row, 0].set_title(
            f"original\ntrue={true_names[row]}\npred={original_pred_names[row]}"
        )
        axes[row, 0].axis("off")

        axes[row, 1].imshow(original_np)
        axes[row, 1].imshow(mask_cpu[row, 0].numpy(), cmap="magma", alpha=0.45)
        axes[row, 1].set_title("background mask")
        axes[row, 1].axis("off")

        for col, method_name in enumerate(method_names, start=2):
            image_np = perturbed_denorm[method_name][row].permute(1, 2, 0).numpy()
            axes[row, col].imshow(image_np)
            axes[row, col].set_title(
                f"{method_name}\npred={perturbed_pred_names[method_name][row]}"
            )
            axes[row, col].axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("saved perturbation grid: %s", output_path)


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Log compact tensor statistics for perturbation sanity checks."""
    detached = tensor.detach()
    LOGGER.info(
        "%s shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
        name,
        tuple(detached.shape),
        detached.min().item(),
        detached.max().item(),
        detached.float().mean().item(),
        detached.float().std().item(),
    )
