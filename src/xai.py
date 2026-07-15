"""Attribution utilities for the AwA2 explainability audit."""

from __future__ import annotations

import logging
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from captum.attr import GradientShap, IntegratedGradients, LayerAttribution, LayerGradCam, Saliency
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


def normalize_channel_maps(maps: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Min-max normalize every channel of every map independently.

    Score-CAM uses each activation channel as a separate input mask. Sharing
    one min/max range across channels would suppress low-amplitude channels
    before their masked class scores are measured.
    """
    if maps.dim() != 4:
        raise ValueError("Expected channel maps with shape [B, C, H, W].")
    flat = maps.flatten(start_dim=2)
    mins = flat.min(dim=2).values.unsqueeze(-1).unsqueeze(-1)
    maxs = flat.max(dim=2).values.unsqueeze(-1).unsqueeze(-1)
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


class ScoreCAM:
    """Score-CAM implementation for one convolutional layer.

    Captum does not provide Score-CAM, so this remains the one local attribution
    method kept for parity with the report artifacts.
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: nn.Module,
        max_channels: int | None = 64,
        batch_size: int = 16,
        blur_radius: float = 18.0,
    ) -> None:
        if max_channels is not None and max_channels < 1:
            raise ValueError("max_channels must be positive or None.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        if not math.isfinite(blur_radius) or blur_radius < 0.0:
            raise ValueError("blur_radius must be a non-negative finite number.")
        self.model = model
        self.target_layer = target_layer
        self.max_channels = max_channels
        self.batch_size = batch_size
        self.blur_radius = blur_radius
        self.activations: torch.Tensor | None = None
        self._forward_handle = target_layer.register_forward_hook(self._save_activations)

    def close(self) -> None:
        self._forward_handle.remove()

    def _save_activations(
        self,
        _module: nn.Module,
        _inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
        self.activations = output.detach()

    def __call__(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Return normalized Score-CAM maps with shape [B, 1, H, W]."""
        self.model.eval()
        with torch.no_grad():
            _ = self.model(inputs)

        if self.activations is None:
            raise RuntimeError("Score-CAM hook did not capture activations.")

        activations = F.relu(self.activations)
        baseline = blurred_baseline(inputs, blur_radius=self.blur_radius)
        maps: list[torch.Tensor] = []
        selected_weights: list[torch.Tensor] = []

        for index in range(inputs.size(0)):
            sample_activations = activations[index : index + 1]
            channel_scores = sample_activations.flatten(start_dim=2).amax(dim=2).squeeze(0)
            if self.max_channels is not None and self.max_channels < channel_scores.numel():
                channel_indices = torch.topk(channel_scores, k=self.max_channels).indices
                sample_activations = sample_activations[:, channel_indices]
            else:
                channel_indices = torch.arange(channel_scores.numel(), device=inputs.device)

            masks = F.interpolate(
                sample_activations,
                size=inputs.shape[-2:],
                mode="bilinear",
                align_corners=False,
            )
            masks = normalize_channel_maps(masks)
            masked_inputs = baseline[index : index + 1] + masks.transpose(0, 1) * (
                inputs[index : index + 1] - baseline[index : index + 1]
            )

            weights: list[torch.Tensor] = []
            for start in range(0, masked_inputs.size(0), self.batch_size):
                batch = masked_inputs[start : start + self.batch_size]
                probabilities = torch.softmax(self.model(batch), dim=1)
                target = int(targets[index].item())
                weights.append(probabilities[:, target].detach())
            channel_weights = torch.cat(weights, dim=0).view(1, -1, 1, 1)

            cam = (channel_weights * sample_activations).sum(dim=1, keepdim=True)
            cam = F.relu(cam)
            cam = F.interpolate(cam, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
            maps.append(cam)
            selected_weights.append(channel_weights.flatten())

            LOGGER.info(
                "scorecam.sample index=%d target=%d selected_channels=%d first_channel=%d",
                index,
                int(targets[index].item()),
                int(sample_activations.size(1)),
                int(channel_indices[0].item()) if channel_indices.numel() else -1,
            )

        scorecam_maps = normalize_maps(torch.cat(maps, dim=0))
        weights_for_log = torch.nn.utils.rnn.pad_sequence(
            [weights.detach().cpu() for weights in selected_weights],
            batch_first=True,
            padding_value=0.0,
        )

        log_tensor_stats("scorecam.activations", activations)
        log_tensor_stats("scorecam.weights", weights_for_log)
        log_tensor_stats("scorecam.map", scorecam_maps)
        return scorecam_maps.detach()


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
    """Compute IG attribution maps using the blurred-image baseline."""
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


def _prepare_expected_gradient_baselines(
    inputs: torch.Tensor,
    baselines: torch.Tensor | None,
) -> torch.Tensor:
    if baselines is None:
        raise ValueError(
            "Expected Gradients requires an explicit reference pool, such as "
            "normalized images collected from the training split."
        )
    baselines = baselines.detach().to(device=inputs.device, dtype=inputs.dtype)
    if baselines.dim() != 4 or baselines.size(1) != inputs.size(1):
        raise ValueError("Expected baselines with shape [N, C, H, W].")
    if baselines.size(0) < 2:
        raise ValueError("Expected Gradients requires at least two reference baselines.")
    if not torch.isfinite(baselines).all():
        raise ValueError("Expected Gradients baselines must contain only finite values.")
    if baselines.shape[-2:] != inputs.shape[-2:]:
        baselines = F.interpolate(baselines, size=inputs.shape[-2:], mode="bilinear", align_corners=False)
    return baselines


def _batched_gradient_shap_attribute(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    baseline_pool: torch.Tensor,
    n_samples: int,
    internal_batch_size: int | None,
) -> torch.Tensor:
    """Run GradientShap while bounding expanded samples per forward pass."""
    if inputs.size(0) == 0:
        raise ValueError("Expected Gradients inputs must not be empty.")
    if targets.dim() != 1 or targets.size(0) != inputs.size(0):
        raise ValueError("targets must have shape [B] matching the inputs batch.")

    gradient_shap = GradientShap(model)
    if internal_batch_size is None:
        return gradient_shap.attribute(
            inputs,
            baselines=baseline_pool,
            target=targets,
            n_samples=n_samples,
            stdevs=0.0,
        )
    if internal_batch_size < 1:
        raise ValueError("internal_batch_size must be positive or None.")

    attribution_chunks: list[torch.Tensor] = []
    input_chunk_size = min(inputs.size(0), internal_batch_size)
    for start in range(0, inputs.size(0), input_chunk_size):
        input_chunk = inputs[start : start + input_chunk_size]
        target_chunk = targets[start : start + input_chunk_size]
        samples_per_call = max(1, internal_batch_size // input_chunk.size(0))
        weighted_attributions = torch.zeros_like(input_chunk)
        completed_samples = 0

        while completed_samples < n_samples:
            call_samples = min(samples_per_call, n_samples - completed_samples)
            call_attributions = gradient_shap.attribute(
                input_chunk,
                baselines=baseline_pool,
                target=target_chunk,
                n_samples=call_samples,
                stdevs=0.0,
            )
            weighted_attributions += call_attributions * call_samples
            completed_samples += call_samples

        attribution_chunks.append(weighted_attributions / n_samples)

    return torch.cat(attribution_chunks, dim=0)


def expected_gradients_maps(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    baselines: torch.Tensor | None = None,
    n_samples: int = 24,
    internal_batch_size: int | None = 8,
    seed: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute Expected Gradients from an explicit reference distribution."""
    if n_samples < 1:
        raise ValueError("n_samples must be at least 1.")

    model.eval()
    baseline_pool = _prepare_expected_gradient_baselines(inputs, baselines)
    rng_devices = [inputs.device.index or 0] if inputs.device.type == "cuda" else []
    numpy_rng_state = np.random.get_state() if seed is not None else None
    try:
        with torch.random.fork_rng(devices=rng_devices, enabled=seed is not None):
            if seed is not None:
                # Captum draws both reference indices and interpolation factors
                # through NumPy, while NoiseTunnel samples through PyTorch.
                np.random.seed(seed)
                torch.manual_seed(seed)
                if inputs.device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
            attributions = _batched_gradient_shap_attribute(
                model=model,
                inputs=inputs,
                targets=targets,
                baseline_pool=baseline_pool,
                n_samples=n_samples,
                internal_batch_size=internal_batch_size,
            )
    finally:
        if numpy_rng_state is not None:
            np.random.set_state(numpy_rng_state)
    maps = attributions_to_saliency_map(attributions)

    log_tensor_stats("expected_gradients.baseline_pool", baseline_pool)
    log_tensor_stats("expected_gradients.attributions", attributions)
    log_tensor_stats("expected_gradients.map", maps)
    return maps.detach(), attributions.detach(), baseline_pool.detach()


def expected_gradients(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    baselines: torch.Tensor | None = None,
    n_samples: int = 24,
    internal_batch_size: int | None = 8,
    seed: int | None = None,
) -> torch.Tensor:
    """Return only normalized Expected Gradients maps for metric pipelines."""
    maps, _attributions, _baseline_pool = expected_gradients_maps(
        model=model,
        inputs=inputs,
        targets=targets,
        baselines=baselines,
        n_samples=n_samples,
        internal_batch_size=internal_batch_size,
        seed=seed,
    )
    return maps


def input_gradient_saliency(
    model: nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    """Compute input-gradient saliency maps through Captum Saliency."""
    model.zero_grad(set_to_none=True)
    gradient_inputs = inputs.detach().clone().requires_grad_(True)
    attributions = Saliency(model).attribute(gradient_inputs, target=targets, abs=True)
    maps = attributions_to_saliency_map(attributions)
    log_tensor_stats("vanilla_gradients.raw", attributions)
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
    scorecam_maps: torch.Tensor | None = None,
    expected_gradients_maps: torch.Tensor | None = None,
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
    method_columns: list[tuple[str, torch.Tensor]] = []
    if input_gradient_maps is not None:
        method_columns.append(("Input gradients", input_gradient_maps))
    method_columns.append(("Grad-CAM", gradcam_maps))
    if scorecam_maps is not None:
        method_columns.append(("Score-CAM", scorecam_maps))
    method_columns.append(("Integrated Gradients\nblurred baseline", ig_maps))
    if expected_gradients_maps is not None:
        method_columns.append(("Expected Gradients\nbaseline distribution", expected_gradients_maps))

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
