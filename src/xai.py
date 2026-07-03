"""Grad-CAM and Integrated Gradients utilities."""

from __future__ import annotations

import logging
import gc
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import IntegratedGradients
from PIL import Image, ImageFilter
from torch import nn
from torchvision.transforms import functional as TF

from src.data import IMAGENET_MEAN, IMAGENET_STD, denormalize_batch

LOGGER = logging.getLogger(__name__)


class GradCAM:
    """Minimal Grad-CAM implementation for a target convolutional layer."""

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

    def _save_activations(self, _module: nn.Module, _inputs: tuple[torch.Tensor], output: torch.Tensor) -> None:
        self.activations = output.detach()

    def _save_gradients(
        self,
        _module: nn.Module,
        _grad_input: tuple[torch.Tensor],
        grad_output: tuple[torch.Tensor],
    ) -> None:
        self.gradients = grad_output[0].detach()

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return normalized Grad-CAM maps with shape [B, 1, H, W]."""
        self.model.zero_grad(set_to_none=True)
        logits = self.model(inputs)
        selected = logits.gather(1, targets.view(-1, 1)).sum()
        selected.backward()

        if self.activations is None or self.gradients is None:
            raise RuntimeError("Grad-CAM hooks did not capture activations/gradients.")

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        cam = (weights * self.activations).sum(dim=1, keepdim=True)
        cam = F.relu(cam)
        cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
        cam = normalize_maps(cam)

        log_tensor_stats("gradcam.activations", self.activations)
        log_tensor_stats("gradcam.gradients", self.gradients)
        log_tensor_stats("gradcam.map", cam)
        return cam.detach()


def normalize_maps(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize saliency maps per sample."""
    flat = maps.flatten(start_dim=1)
    mins = flat.min(dim=1).values.view(-1, 1, 1, 1)
    maxs = flat.max(dim=1).values.view(-1, 1, 1, 1)
    return (maps - mins) / (maxs - mins + eps)


def blurred_baseline(inputs: torch.Tensor, kernel_size: int = 51, sigma: float = 18.0) -> torch.Tensor:
    """Create an IG baseline by heavily blurring the denormalized image.

    The baseline is blurred in image space, then normalized back into ResNet
    input space. This preserves average brightness better than a black image.
    """
    device = inputs.device
    denorm = denormalize_batch(inputs.detach()).clamp(0, 1).cpu()
    blurred_tensors: list[torch.Tensor] = []
    radius = max(float(sigma), float(kernel_size) / 6.0)

    for image in denorm:
        pil = TF.to_pil_image(image)
        blurred = pil.filter(ImageFilter.GaussianBlur(radius=radius))
        blurred_tensors.append(TF.to_tensor(blurred))

    baseline = torch.stack(blurred_tensors, dim=0).to(device)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    baseline = (baseline - mean) / std
    log_tensor_stats("ig.blurred_baseline", baseline)
    return baseline


def black_baseline(inputs: torch.Tensor) -> torch.Tensor:
    """Create an all-black baseline in normalized ResNet input space."""
    device = inputs.device
    baseline = torch.zeros_like(inputs)
    mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)
    baseline = (baseline - mean) / std
    log_tensor_stats("ig.black_baseline", baseline)
    return baseline


def input_gradient_maps(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute vanilla input-gradient saliency maps.

    The raw gradient has shape [B, 3, H, W] and keeps the RGB channel-level
    sensitivity. The returned map is the normalized absolute channel sum with
    shape [B, 1, H, W].
    """
    gradient_inputs = inputs.detach().clone().requires_grad_(True)
    model.zero_grad(set_to_none=True)
    logits = model(gradient_inputs)
    selected = logits.gather(1, targets.view(-1, 1)).sum()
    selected.backward()

    if gradient_inputs.grad is None:
        raise RuntimeError("Input gradients were not computed.")

    gradients = gradient_inputs.grad.detach()
    maps = normalize_maps(gradients.abs().sum(dim=1, keepdim=True))
    log_tensor_stats("vanilla_gradients.raw", gradients)
    log_tensor_stats("vanilla_gradients.map", maps)
    model.zero_grad(set_to_none=True)
    return maps.detach(), gradients.detach()


def integrated_gradients_maps(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    steps: int = 50,
    internal_batch_size: int | None = 4,
    baseline_type: str = "blurred",
) -> torch.Tensor:
    """Compute normalized Integrated Gradients attribution maps."""
    ig = IntegratedGradients(model)
    if baseline_type == "blurred":
        baseline = blurred_baseline(inputs)
    elif baseline_type == "black":
        baseline = black_baseline(inputs)
    else:
        raise ValueError(f"Unsupported IG baseline_type: {baseline_type}")
    attributions = ig.attribute(
        inputs,
        baselines=baseline,
        target=targets,
        n_steps=steps,
        internal_batch_size=internal_batch_size,
    )
    maps = attributions.abs().sum(dim=1, keepdim=True)
    maps = normalize_maps(maps)
    log_tensor_stats("ig.attributions", attributions)
    log_tensor_stats("ig.map", maps)
    del baseline
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return maps.detach()


def saliency_concentration(
    saliency_maps: torch.Tensor,
    mask_paths: list[str],
    foreground_value: int = 1,
    background_value: int = 2,
) -> list[dict[str, float]]:
    """Measure saliency mass inside foreground/background trimap regions."""
    maps = saliency_maps.detach().cpu()
    metrics: list[dict[str, float]] = []

    for idx, mask_path in enumerate(mask_paths):
        mask_image = Image.open(mask_path)
        mask_image = TF.resize(
            mask_image,
            size=list(maps.shape[-2:]),
            interpolation=TF.InterpolationMode.NEAREST,
        )
        mask = TF.pil_to_tensor(mask_image).squeeze(0)
        saliency = maps[idx, 0]
        total = saliency.sum().item() + 1e-8
        foreground = saliency[mask == foreground_value].sum().item() / total
        background = saliency[mask == background_value].sum().item() / total
        boundary = max(0.0, 1.0 - foreground - background)
        metrics.append(
            {
                "foreground_saliency_mass": foreground,
                "background_saliency_mass": background,
                "boundary_saliency_mass": boundary,
            }
        )
    return metrics


def saliency_iou_at_percentile(
    maps_a: torch.Tensor,
    maps_b: torch.Tensor,
    percentile: float = 80.0,
) -> torch.Tensor:
    """IoU between top-saliency binary masks from two map tensors."""
    flat_a = maps_a.flatten(start_dim=1)
    flat_b = maps_b.flatten(start_dim=1)
    threshold_a = torch.quantile(flat_a, percentile / 100.0, dim=1).view(-1, 1, 1, 1)
    threshold_b = torch.quantile(flat_b, percentile / 100.0, dim=1).view(-1, 1, 1, 1)
    mask_a = maps_a >= threshold_a
    mask_b = maps_b >= threshold_b
    intersection = (mask_a & mask_b).flatten(start_dim=1).sum(dim=1).float()
    union = (mask_a | mask_b).flatten(start_dim=1).sum(dim=1).float().clamp_min(1.0)
    return intersection / union


def spearman_rank_correlation(maps_a: torch.Tensor, maps_b: torch.Tensor) -> torch.Tensor:
    """Spearman rank correlation for flattened saliency maps."""
    flat_a = maps_a.flatten(start_dim=1)
    flat_b = maps_b.flatten(start_dim=1)
    ranks_a = torch.argsort(torch.argsort(flat_a, dim=1), dim=1).float()
    ranks_b = torch.argsort(torch.argsort(flat_b, dim=1), dim=1).float()
    ranks_a = ranks_a - ranks_a.mean(dim=1, keepdim=True)
    ranks_b = ranks_b - ranks_b.mean(dim=1, keepdim=True)
    numerator = (ranks_a * ranks_b).sum(dim=1)
    denominator = ranks_a.norm(dim=1) * ranks_b.norm(dim=1)
    return numerator / denominator.clamp_min(1e-8)


def save_multi_xai_grid(
    images: torch.Tensor,
    vanilla_maps: torch.Tensor,
    gradcam_maps: torch.Tensor,
    ig_blur_maps: torch.Tensor,
    ig_black_maps: torch.Tensor,
    class_names: list[str],
    predictions: list[str],
    confidences: list[float],
    output_path: str | Path,
) -> None:
    """Save a multi-example grid comparing Grad-CAM and IG baselines."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    denorm = denormalize_batch(images.detach().cpu()).clamp(0, 1)
    n = images.shape[0]
    fig, axes = plt.subplots(n, 5, figsize=(16, 3.1 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    titles = ["image", "vanilla gradients", "Grad-CAM", "IG blurred baseline", "IG black baseline"]
    for row in range(n):
        image_np = denorm[row].permute(1, 2, 0).numpy()
        maps = [
            None,
            vanilla_maps[row, 0].detach().cpu().numpy(),
            gradcam_maps[row, 0].detach().cpu().numpy(),
            ig_blur_maps[row, 0].detach().cpu().numpy(),
            ig_black_maps[row, 0].detach().cpu().numpy(),
        ]
        for col in range(5):
            if col == 0:
                axes[row, col].imshow(image_np)
                axes[row, col].set_title(
                    f"{titles[col]}\ntrue={class_names[row]}\npred={predictions[row]} ({confidences[row]:.2f})"
                )
            else:
                axes[row, col].imshow(overlay_heatmap(image_np, maps[col]))
                axes[row, col].set_title(titles[col])
            axes[row, col].axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved multi-XAI grid: %s", output_path)


def write_xai_metrics_csv(rows: list[dict[str, object]], output_path: str | Path) -> None:
    """Write XAI metric rows to CSV."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError("No metric rows to write.")
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    LOGGER.info("Saved XAI metrics: %s", output_path)


def overlay_heatmap(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.45) -> np.ndarray:
    """Overlay a saliency heatmap on an RGB image in [0, 1]."""
    cmap = plt.get_cmap("jet")
    colored = cmap(heatmap)[..., :3]
    overlay = (1.0 - alpha) * image + alpha * colored
    return np.clip(overlay, 0.0, 1.0)


def save_xai_grid(
    images: torch.Tensor,
    gradcam_maps: torch.Tensor,
    ig_maps: torch.Tensor,
    class_names: list[str],
    predictions: list[str],
    confidences: list[float],
    output_path: str | Path,
) -> None:
    """Save original/Grad-CAM/IG comparison grid."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    denorm = denormalize_batch(images.detach().cpu()).clamp(0, 1)
    n = images.shape[0]
    fig, axes = plt.subplots(n, 3, figsize=(10, 3.2 * n))
    if n == 1:
        axes = np.expand_dims(axes, axis=0)

    for idx in range(n):
        image_np = denorm[idx].permute(1, 2, 0).numpy()
        gradcam_np = gradcam_maps[idx, 0].detach().cpu().numpy()
        ig_np = ig_maps[idx, 0].detach().cpu().numpy()

        axes[idx, 0].imshow(image_np)
        axes[idx, 0].set_title(f"image\ntrue={class_names[idx]}")
        axes[idx, 1].imshow(overlay_heatmap(image_np, gradcam_np))
        axes[idx, 1].set_title(f"Grad-CAM\npred={predictions[idx]} ({confidences[idx]:.2f})")
        axes[idx, 2].imshow(overlay_heatmap(image_np, ig_np))
        axes[idx, 2].set_title("Integrated Gradients")
        for axis in axes[idx]:
            axis.axis("off")

    plt.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    LOGGER.info("Saved XAI grid: %s", output_path)


def log_tensor_stats(name: str, tensor: torch.Tensor) -> None:
    """Log compact tensor statistics for XAI sanity checks."""
    tensor_detached = tensor.detach()
    LOGGER.info(
        "%s shape=%s min=%.6f max=%.6f mean=%.6f std=%.6f",
        name,
        tuple(tensor_detached.shape),
        tensor_detached.min().item(),
        tensor_detached.max().item(),
        tensor_detached.float().mean().item(),
        tensor_detached.float().std().item(),
    )
