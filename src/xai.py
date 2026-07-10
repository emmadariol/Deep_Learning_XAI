"""Grad-CAM and Integrated Gradients utilities for AwA2."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from PIL import ImageFilter
from torch import nn
from torchvision.transforms import functional as TF

from src.data import IMAGENET_MEAN, IMAGENET_STD, denormalize_batch

LOGGER = logging.getLogger(__name__)


class GradCAM:
    """Small explicit Grad-CAM implementation for one convolutional layer."""

    def __init__(self, model: nn.Module, target_layer: nn.Module) -> None:
        self.model = model
        self.target_layer = target_layer
        self.activations: torch.Tensor | None = None
        self.gradients: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)
        self._backward_handle = target_layer.register_full_backward_hook(self._save_gradients)

    def close(self) -> None:
        self._forward_handle.remove()
        self._backward_handle.remove()

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
        """Return normalized Grad-CAM maps with shape [B, 1, H, W]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        selected_scores = logits.gather(1, targets.view(-1, 1)).sum()
        selected_scores.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        cam = normalize_maps(cam)

        log_tensor_stats("gradcam.activations", self.activations)
        log_tensor_stats("gradcam.gradients", self.gradients)
        log_tensor_stats("gradcam.weights", weights)
        log_tensor_stats("gradcam.map", cam)
        return cam.detach()


def normalize_maps(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize each saliency map independently to [0, 1]."""
    flat = maps.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (maps - mins) / (maxs - mins + eps)


def blurred_baseline(
    inputs: torch.Tensor,
    blur_radius: float = 18.0,
) -> torch.Tensor:
    """Create a blurred-image baseline for Integrated Gradients.

    The blur is applied in RGB image space, not normalized tensor space. This
    preserves average brightness and color statistics better than a black image.
    """
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
    """Compute IG attribution maps using the blurred-image baseline."""
    baseline = blurred_baseline(inputs, blur_radius=blur_radius)
    ig = IntegratedGradients(model)
    attributions = ig.attribute(
        inputs,
        baselines=baseline,
        target=targets,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
    )
    maps = normalize_maps(attributions.abs().sum(dim=1, keepdim=True))

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
    """Compute normalized input-gradient saliency maps with shape [B, 1, H, W]."""
    model.zero_grad(set_to_none=True)
    gradient_inputs = inputs.detach().clone().requires_grad_(True)
    logits = model(gradient_inputs)
    selected_scores = logits.gather(1, targets.view(-1, 1)).sum()
    selected_scores.backward()

    if gradient_inputs.grad is None:
        raise RuntimeError("Input gradients were not computed.")

    gradients = gradient_inputs.grad.detach()
    maps = normalize_maps(gradients.abs().sum(dim=1, keepdim=True))
    log_tensor_stats("vanilla_gradients.raw", gradients)
    log_tensor_stats("vanilla_gradients.map", maps)
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
    input_gradient_maps: torch.Tensor | None = None,
    predicted_names: list[str] | None = None,
) -> None:
    """Save image and XAI overlays.

    Supports both the older 3-column API and the richer notebook API that also
    passes input-gradient saliency maps and ``predicted_names``.
    """
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
    has_input_gradients = input_gradient_maps is not None
    n_columns = 4 if has_input_gradients else 3
    fig, axes = plt.subplots(n_images, n_columns, figsize=(3.4 * n_columns, 3.2 * n_images))
    if n_images == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(n_images):
        image_np = denorm[idx].permute(1, 2, 0).numpy()
        gradcam_np = gradcam_maps[idx, 0].detach().cpu().numpy()
        ig_np = ig_maps[idx, 0].detach().cpu().numpy()

        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f"image\ntrue={true_names[idx]}")
        next_col = 1

        if has_input_gradients:
            input_gradient_np = input_gradient_maps[idx, 0].detach().cpu().numpy()
            axes[idx, next_col].imshow(overlay_heatmap(image_np, input_gradient_np))
            axes[idx, next_col].set_title("Input gradients")
            next_col += 1

        axes[idx, next_col].imshow(overlay_heatmap(image_np, gradcam_np))
        axes[idx, next_col].set_title(f"Grad-CAM\npred={pred_names[idx]} ({confidences[idx]:.2f})")
        next_col += 1

        axes[idx, next_col].imshow(overlay_heatmap(image_np, ig_np))
        axes[idx, next_col].set_title("Integrated Gradients\nblurred baseline")

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
