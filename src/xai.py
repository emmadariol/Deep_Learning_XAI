"""Explicit-gradient XAI methods for AwA2 images."""

from __future__ import annotations

import logging
import os
from pathlib import Path

_CACHE_DIR = Path.cwd() / "outputs" / "cache"
_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_CACHE_DIR / "matplotlib"))
os.environ.setdefault("XDG_CACHE_HOME", str(_CACHE_DIR / "xdg"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from src.data import IMAGENET_MEAN, IMAGENET_STD, denormalize_batch

LOGGER = logging.getLogger(__name__)


def normalize_maps(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each saliency map independently."""
    flat = maps.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (maps - mins) / (maxs - mins + eps)


def input_gradient_saliency(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute saliency from the explicit input gradient d score / d image."""
    gradient_inputs = inputs.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)
    logits = model(gradient_inputs)
    selected_scores = logits.gather(1, targets.view(-1, 1)).sum()
    selected_scores.backward()

    if gradient_inputs.grad is None:
        raise RuntimeError("Expected input gradients, but gradient_inputs.grad is None.")

    gradients = gradient_inputs.grad.detach()
    saliency = gradients.abs().sum(dim=1, keepdim=True)
    LOGGER.info(
        "input gradient: raw_shape=%s saliency_shape=%s grad_min=%.6f grad_max=%.6f",
        tuple(gradients.shape),
        tuple(saliency.shape),
        gradients.min().item(),
        gradients.max().item(),
    )
    return normalize_maps(saliency)


class GradCAM:
    """Minimal Grad-CAM implementation for one convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self.forward_handle = target_layer.register_forward_hook(self._save_activations)
        self.backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def close(self) -> None:
        self.forward_handle.remove()
        self.backward_handle.remove()

    def _save_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor, ...],
        grad_output: tuple[torch.Tensor, ...],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        selected_scores = logits.gather(1, targets.view(-1, 1)).sum()
        selected_scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations and gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        LOGGER.info(
            "grad-cam: activations=%s gradients=%s cam=%s",
            tuple(self.activations.shape),
            tuple(self.gradients.shape),
            tuple(cam.shape),
        )
        return normalize_maps(cam.detach())


def blurred_baseline(inputs: torch.Tensor, kernel_size: int = 31) -> torch.Tensor:
    """Create a blurred baseline in normalized input space."""
    if kernel_size % 2 == 0:
        raise ValueError("kernel_size must be odd.")

    denormalized = denormalize_batch(inputs.detach()).clamp(0.0, 1.0)
    padding = kernel_size // 2
    blurred = F.avg_pool2d(
        denormalized,
        kernel_size=kernel_size,
        stride=1,
        padding=padding,
    )

    mean = torch.tensor(IMAGENET_MEAN, device=inputs.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=inputs.device).view(1, 3, 1, 1)
    return (blurred - mean) / std


def integrated_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 16,
) -> torch.Tensor:
    """Compute Integrated Gradients with an explicit gradient loop."""
    if steps <= 0:
        raise ValueError("steps must be positive.")

    baseline = blurred_baseline(inputs)
    total_gradients = torch.zeros_like(inputs)

    for alpha in torch.linspace(0.0, 1.0, steps, device=inputs.device):
        interpolated = baseline + alpha * (inputs - baseline)
        interpolated = interpolated.detach().requires_grad_(True)

        model.zero_grad(set_to_none=True)
        logits = model(interpolated)
        selected_scores = logits.gather(1, targets.view(-1, 1)).sum()
        selected_scores.backward()

        if interpolated.grad is None:
            raise RuntimeError("Expected IG gradients, but interpolated.grad is None.")
        total_gradients += interpolated.grad.detach()

    average_gradients = total_gradients / float(steps)
    attributions = (inputs - baseline) * average_gradients
    saliency = attributions.abs().sum(dim=1, keepdim=True)
    LOGGER.info(
        "integrated gradients: steps=%d attribution_shape=%s",
        steps,
        tuple(attributions.shape),
    )
    return normalize_maps(saliency)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    cmap = plt.get_cmap("jet")
    heatmap_rgb = cmap(heatmap)[..., :3]
    return np.clip((1.0 - alpha) * image + alpha * heatmap_rgb, 0.0, 1.0)


def save_xai_grid(
    images: torch.Tensor,
    input_gradient_maps: torch.Tensor,
    gradcam_maps: torch.Tensor,
    ig_maps: torch.Tensor,
    true_names: list[str],
    predicted_names: list[str],
    confidences: list[float],
    output_path: str | Path,
) -> None:
    """Save a compact grid with original image and three attribution maps."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    denormalized = denormalize_batch(images.detach().cpu()).clamp(0.0, 1.0)
    row_count = images.size(0)
    fig, axes = plt.subplots(row_count, 4, figsize=(12, 3.0 * row_count))
    if row_count == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["image", "input gradient", "Grad-CAM", "Integrated Gradients"]
    for row in range(row_count):
        image_np = denormalized[row].permute(1, 2, 0).numpy()
        maps = [
            None,
            input_gradient_maps[row, 0].detach().cpu().numpy(),
            gradcam_maps[row, 0].detach().cpu().numpy(),
            ig_maps[row, 0].detach().cpu().numpy(),
        ]

        for col in range(4):
            axis = axes[row, col]
            if col == 0:
                axis.imshow(image_np)
                axis.set_title(
                    f"{titles[col]}\ntrue={true_names[row]}\n"
                    f"pred={predicted_names[row]} ({confidences[row]:.2f})"
                )
            else:
                axis.imshow(overlay_heatmap(image_np, maps[col]))
                axis.set_title(titles[col])
            axis.axis("off")

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    LOGGER.info("saved XAI grid: %s", output_path)
