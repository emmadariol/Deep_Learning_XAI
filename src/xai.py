"""Attribution utilities for the AwA2 explainability audit.

Maintained local attribution methods:

- Grad-CAM
- Integrated Gradients with a blurred-image baseline

The project intentionally excludes older exploratory attribution methods from
this module so scripts, notebooks and reports expose one coherent method set.
"""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import IntegratedGradients, LayerAttribution, LayerGradCam, Saliency
from PIL import ImageFilter
from torch import nn
from torchvision.transforms import functional as TF

from src.data import IMAGENET_MEAN, IMAGENET_STD, denormalize_batch

LOGGER = logging.getLogger(__name__)


def normalize_maps(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each saliency map independently to [0, 1]."""
    flat = maps.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (maps - mins) / (maxs - mins + eps)


def attributions_to_saliency_map(
    attributions: torch.Tensor,
    positive_only: bool = False,
) -> torch.Tensor:
    """Collapse Captum attributions to normalized single-channel saliency maps."""
    if attributions.dim() != 4:
        raise ValueError("Expected attributions with shape [B, C, H, W].")
    maps = torch.clamp(attributions, min=0.0) if positive_only else attributions.abs()
    if maps.size(1) != 1:
        maps = maps.sum(dim=1, keepdim=True)
    return normalize_maps(maps)


def gradcam_saliency(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    target_layer: nn.Module | None = None,
) -> torch.Tensor:
    """Compute Grad-CAM maps through Captum LayerGradCam."""
    layer = target_layer if target_layer is not None else model.layer4[-1]
    model.zero_grad(set_to_none=True)
    gradient_inputs = inputs.detach().clone().requires_grad_(True)
    attributions = LayerGradCam(model, layer).attribute(
        gradient_inputs,
        target=targets,
        relu_attributions=True,
    )
    upsampled = LayerAttribution.interpolate(
        attributions,
        inputs.shape[-2:],
        interpolate_mode="bilinear",
    )
    maps = attributions_to_saliency_map(upsampled, positive_only=True)
    log_tensor_stats("gradcam.map", maps)
    return maps.detach()


def blurred_baseline(
    inputs: torch.Tensor,
    blur_radius: float = 18.0,
) -> torch.Tensor:
    """Create a blurred-image baseline for Integrated Gradients.

    The blur is applied in RGB image space, not normalized tensor space. This
    preserves average brightness and color statistics better than a black image.
    """
    if not math.isfinite(blur_radius) or blur_radius < 0.0:
        raise ValueError("blur_radius must be a non-negative finite number.")
    if inputs.dim() != 4 or inputs.size(0) == 0 or inputs.size(1) != 3:
        raise ValueError("inputs must contain at least one RGB image with shape [B, 3, H, W].")
    device = inputs.device
    denorm = denormalize_batch(inputs.detach()).clamp(0, 1).cpu()
    blurred_tensors: list[torch.Tensor] = []

    for image in denorm:
        pil_image = TF.to_pil_image(image)
        blurred = pil_image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        blurred_tensors.append(TF.to_tensor(blurred))

    baseline = torch.stack(blurred_tensors, dim=0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    baseline = (baseline - mean) / std
    log_tensor_stats("ig.blurred_baseline", baseline)
    return baseline


def integrated_gradients_maps(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 50,
    internal_batch_size: int | None = 4,
    blur_radius: float = 18.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Integrated Gradients using a blurred-image baseline."""
    if steps < 1:
        raise ValueError("steps must be positive.")
    if internal_batch_size is not None and internal_batch_size < 1:
        raise ValueError("internal_batch_size must be positive or None.")
    baseline = blurred_baseline(inputs, blur_radius=blur_radius)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs,
        baselines=baseline,
        target=targets,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
    )
    maps = attributions_to_saliency_map(attributions)

    log_tensor_stats("ig.attributions", attributions)
    log_tensor_stats("ig.map", maps)
    return maps.detach(), attributions.detach(), baseline.detach()


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 50,
    internal_batch_size: int | None = 4,
    blur_radius: float = 18.0,
) -> torch.Tensor:
    """Return only normalized Integrated Gradients maps for metric pipelines."""
    maps, _attributions, _baseline = integrated_gradients_maps(
        model=model,
        inputs=inputs,
        targets=targets,
        steps=steps,
        internal_batch_size=internal_batch_size,
        blur_radius=blur_radius,
    )
    return maps


def input_gradient_saliency(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute plain input-gradient saliency maps for debugging only."""
    model.zero_grad(set_to_none=True)
    gradient_inputs = inputs.detach().clone().requires_grad_(True)
    attributions = Saliency(model).attribute(gradient_inputs, target=targets, abs=True)
    maps = attributions_to_saliency_map(attributions)
    log_tensor_stats("input_gradients.raw", attributions)
    log_tensor_stats("input_gradients.map", maps)
    return maps.detach()


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a normalized heatmap on an RGB image in [0, 1]."""
    colored = plt.get_cmap("jet")(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * colored, 0.0, 1.0)


def save_xai_grid(
    images: torch.Tensor,
    gradcam_maps: torch.Tensor,
    ig_maps: torch.Tensor,
    true_names: list[str],
    pred_names: list[str] | None = None,
    confidences: list[float] | None = None,
    output_path: str | Path | None = None,
    predicted_names: list[str] | None = None,
) -> None:
    """Save original images with Grad-CAM and Integrated Gradients overlays."""
    if pred_names is None:
        pred_names = predicted_names
    if pred_names is None:
        raise ValueError("pred_names or predicted_names must be provided.")
    if confidences is None:
        confidences = [float("nan")] * len(pred_names)
    if output_path is None:
        raise ValueError("output_path must be provided.")

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    denorm = denormalize_batch(images.detach().cpu()).clamp(0, 1)
    n_images = images.shape[0]
    method_columns: list[tuple[str, torch.Tensor]] = [
        ("Grad-CAM", gradcam_maps),
        ("Integrated Gradients\nblurred baseline", ig_maps),
    ]

    n_columns = 1 + len(method_columns)
    fig, axes = plt.subplots(n_images, n_columns, figsize=(3.4 * n_columns, 3.2 * n_images))
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(n_images):
        image_np = denorm[idx].permute(1, 2, 0).numpy()
        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(
            f"image\ntrue={true_names[idx]}\npred={pred_names[idx]} ({confidences[idx]:.2f})"
        )

        for offset, (title, maps) in enumerate(method_columns, start=1):
            map_np = maps[idx, 0].detach().cpu().numpy()
            axes[idx, offset].imshow(overlay_heatmap(image_np, map_np))
            axes[idx, offset].set_title(title)

        for axis in axes[idx]:
            axis.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved XAI grid: %s", output_path)


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Log compact tensor statistics for normalization/debug checks."""
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
