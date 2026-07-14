"""Forward-pass inspection utilities for the ResNet50 AwA2 classifier."""

from __future__ import annotations

import base64
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

from src.data import denormalize_batch
from src.xai import GradCAM, normalize_maps, overlay_heatmap

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TensorStats:
    """Compact numerical summary of one intermediate tensor."""

    name: str
    shape: tuple[int, ...]
    minimum: float
    maximum: float
    mean: float
    std: float


@dataclass
class PredictionTrace:
    """Forward-pass trace for one image."""

    logits: torch.Tensor
    probabilities: torch.Tensor
    predicted_label: int
    confidence: float
    activations: dict[str, torch.Tensor]
    activation_maps: dict[str, torch.Tensor]
    gradcam_map: torch.Tensor | None
    stats: list[TensorStats]


class ForwardActivationInspector:
    """Capture real intermediate tensors from selected ResNet50 modules."""

    def __init__(
        self,
        model: nn.Module,
        layer_names: Iterable[str] = (
            "conv1",
            "maxpool",
            "layer1",
            "layer2",
            "layer3",
            "layer4",
            "avgpool",
        ),
    ) -> None:
        self.model = model
        self.layer_names = tuple(layer_names)
        self.activations: dict[str, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def __enter__(self) -> "ForwardActivationInspector":
        for name in self.layer_names:
            module = getattr(self.model, name, None)
            if module is None:
                raise ValueError(f"Model has no top-level module named {name!r}.")
            self._handles.append(module.register_forward_hook(self._make_hook(name)))
        return self

    def __exit__(self, _exc_type, _exc, _traceback) -> None:
        self.close()

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _make_hook(self, name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            self.activations[name] = output.detach().cpu()

        return hook

    def run(
        self,
        image: torch.Tensor,
        target_label: int | None = None,
        compute_gradcam: bool = True,
    ) -> PredictionTrace:
        """Run one normalized image through the model and capture real tensors."""
        if image.dim() == 3:
            image = image.unsqueeze(0)
        if image.size(0) != 1:
            raise ValueError("ForwardActivationInspector expects exactly one image.")

        device = next(self.model.parameters()).device
        image = image.to(device)
        self.activations.clear()
        self.model.eval()

        with torch.no_grad():
            logits = self.model(image)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predicted = probabilities.max(dim=1)

        target = int(predicted.item() if target_label is None else target_label)
        gradcam_map: torch.Tensor | None = None
        if compute_gradcam:
            gradcam = GradCAM(self.model, self.model.layer4[-1])
            try:
                gradcam_map = gradcam(image, torch.tensor([target], device=device)).detach().cpu()
            finally:
                gradcam.close()

        activation_maps = {
            name: activation_to_map(tensor, image_size=image.shape[-2:])
            for name, tensor in self.activations.items()
            if tensor.dim() == 4
        }
        stats = [
            tensor_stats("input", image.detach().cpu()),
            tensor_stats("logits", logits.detach().cpu()),
            tensor_stats("probabilities", probabilities.detach().cpu()),
        ]
        stats.extend(tensor_stats(name, tensor) for name, tensor in self.activations.items())

        return PredictionTrace(
            logits=logits.detach().cpu(),
            probabilities=probabilities.detach().cpu(),
            predicted_label=int(predicted.item()),
            confidence=float(confidence.item()),
            activations=dict(self.activations),
            activation_maps=activation_maps,
            gradcam_map=gradcam_map,
            stats=stats,
        )


def tensor_stats(name: str, tensor: torch.Tensor) -> TensorStats:
    """Compute stable scalar statistics for a tensor."""
    detached = tensor.detach().float().cpu()
    return TensorStats(
        name=name,
        shape=tuple(detached.shape),
        minimum=float(detached.min().item()),
        maximum=float(detached.max().item()),
        mean=float(detached.mean().item()),
        std=float(detached.std().item()) if detached.numel() > 1 else 0.0,
    )


def activation_to_map(activation: torch.Tensor, image_size: tuple[int, int]) -> torch.Tensor:
    """Convert a captured activation tensor into a normalized spatial map."""
    if activation.dim() != 4:
        raise ValueError(f"Expected [B, C, H, W] activation, got shape {tuple(activation.shape)}.")
    spatial = activation.detach().float().abs().mean(dim=1, keepdim=True)
    spatial = F.interpolate(spatial, size=image_size, mode="bilinear", align_corners=False)
    return normalize_maps(spatial)


def save_prediction_trace_figure(
    image: torch.Tensor,
    trace: PredictionTrace,
    class_names: dict[int, str],
    output_path: str | Path,
    true_label: int | None = None,
    image_path: str | None = None,
    top_k: int = 5,
) -> None:
    """Save a visual summary of a real ResNet50 forward pass."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if image.dim() == 3:
        image = image.unsqueeze(0)
    image_np = denormalize_batch(image.detach().cpu()).clamp(0, 1)[0].permute(1, 2, 0).numpy()
    layer_order = [name for name in ("conv1", "maxpool", "layer1", "layer2", "layer3", "layer4") if name in trace.activation_maps]

    n_cols = 4
    n_rows = 3
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 12))
    flat_axes = axes.flatten()

    predicted_name = class_names.get(trace.predicted_label, str(trace.predicted_label))
    true_name = class_names.get(true_label, str(true_label)) if true_label is not None else "not provided"
    title = f"true={true_name} | predicted={predicted_name} | confidence={trace.confidence:.3f}"
    if image_path:
        title += f"\n{image_path}"
    fig.suptitle(title, fontsize=14)

    flat_axes[0].imshow(image_np)
    flat_axes[0].set_title("Input image\nnormalized RGB -> denormalized view")
    flat_axes[0].axis("off")

    axis_index = 1
    for layer_name in layer_order[:6]:
        map_np = trace.activation_maps[layer_name][0, 0].numpy()
        flat_axes[axis_index].imshow(overlay_heatmap(image_np, map_np, alpha=0.48))
        shape = tuple(trace.activations[layer_name].shape)
        flat_axes[axis_index].set_title(f"{layer_name}\nactivation shape={shape}")
        flat_axes[axis_index].axis("off")
        axis_index += 1

    if trace.gradcam_map is not None and axis_index < len(flat_axes):
        gradcam_np = trace.gradcam_map[0, 0].numpy()
        flat_axes[axis_index].imshow(overlay_heatmap(image_np, gradcam_np, alpha=0.5))
        flat_axes[axis_index].set_title("Grad-CAM\ncomputed on layer4[-1]")
        flat_axes[axis_index].axis("off")
        axis_index += 1

    top_values, top_indices = torch.topk(trace.probabilities[0], k=min(top_k, trace.probabilities.size(1)))
    labels = [class_names.get(int(index), str(int(index))) for index in top_indices]
    values = top_values.numpy()
    flat_axes[axis_index].barh(np.arange(len(labels)), values, color="#315d9c")
    flat_axes[axis_index].set_yticks(np.arange(len(labels)), labels)
    flat_axes[axis_index].invert_yaxis()
    flat_axes[axis_index].set_xlim(0, max(float(values.max()) * 1.15, 0.05))
    flat_axes[axis_index].set_title("Top softmax probabilities")
    flat_axes[axis_index].set_xlabel("probability")
    axis_index += 1

    stats_rows = []
    for stats in trace.stats:
        if stats.name in {"input", "conv1", "maxpool", "layer1", "layer2", "layer3", "layer4", "avgpool", "logits", "probabilities"}:
            stats_rows.append(
                [
                    stats.name,
                    "x".join(str(dim) for dim in stats.shape),
                    f"{stats.minimum:.3f}",
                    f"{stats.maximum:.3f}",
                    f"{stats.mean:.3f}",
                    f"{stats.std:.3f}",
                ]
            )

    flat_axes[axis_index].axis("off")
    table = flat_axes[axis_index].table(
        cellText=stats_rows,
        colLabels=["tensor", "shape", "min", "max", "mean", "std"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 1.35)
    flat_axes[axis_index].set_title("Real tensor diagnostics")
    axis_index += 1

    for index in range(axis_index, len(flat_axes)):
        flat_axes[index].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    LOGGER.info("Saved forward inspection figure: %s", output_path)


def print_trace_summary(trace: PredictionTrace, class_names: dict[int, str], top_k: int = 5) -> None:
    """Print a compact textual trace for notebooks and scripts."""
    predicted_name = class_names.get(trace.predicted_label, str(trace.predicted_label))
    print(f"predicted_label={trace.predicted_label} predicted_name={predicted_name} confidence={trace.confidence:.4f}")

    top_values, top_indices = torch.topk(trace.probabilities[0], k=min(top_k, trace.probabilities.size(1)))
    print("top probabilities:")
    for value, index in zip(top_values.tolist(), top_indices.tolist(), strict=False):
        print(f"  {index:>3} {class_names.get(int(index), str(int(index))):<24} {value:.4f}")

    print("tensor statistics:")
    for stats in trace.stats:
        print(
            f"  {stats.name:<14} shape={stats.shape} "
            f"min={stats.minimum:.5f} max={stats.maximum:.5f} "
            f"mean={stats.mean:.5f} std={stats.std:.5f}"
        )


def _tensor_to_grid(tensor: torch.Tensor, size: tuple[int, int]) -> list:
    """Resize a tensor to a compact JSON-serializable grid."""
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)
    resized = F.interpolate(tensor.detach().float().cpu(), size=size, mode="bilinear", align_corners=False)
    return resized.squeeze(0).tolist()


def _map_to_grid(map_tensor: torch.Tensor, size: tuple[int, int] = (7, 7)) -> list[list[float]]:
    """Resize a [1, 1, H, W] map to a compact 2D grid."""
    resized = F.interpolate(map_tensor.detach().float().cpu(), size=size, mode="bilinear", align_corners=False)
    return resized[0, 0].tolist()


def _image_to_data_url(image: torch.Tensor) -> str:
    """Encode a denormalized RGB tensor as an embedded PNG for browser display."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    denorm = denormalize_batch(image.detach().cpu()).clamp(0, 1)[0]
    image_array = (denorm.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    pil_image = Image.fromarray(image_array, mode="RGB")
    buffer = io.BytesIO()
    pil_image.save(buffer, format="PNG", optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def trace_to_browser_payload(
    image: torch.Tensor,
    trace: PredictionTrace,
    class_names: dict[int, str],
    true_label: int | None = None,
    image_path: str | None = None,
    top_k: int = 10,
) -> dict:
    """Convert a real forward trace into a small JSON payload for the HTML simulator."""
    if image.dim() == 3:
        image = image.unsqueeze(0)
    denorm = denormalize_batch(image.detach().cpu()).clamp(0, 1)
    rgb_grid = _tensor_to_grid(denorm, size=(14, 14))
    rgb_grid = np.transpose(np.array(rgb_grid), (1, 2, 0)).tolist()

    activation_maps: dict[str, list[list[float]]] = {}
    for name, map_tensor in trace.activation_maps.items():
        activation_maps[name] = _map_to_grid(map_tensor, size=(7, 7))

    top_values, top_indices = torch.topk(trace.probabilities[0], k=min(top_k, trace.probabilities.size(1)))
    probabilities = [
        {
            "label": int(index),
            "name": class_names.get(int(index), str(int(index))),
            "probability": float(value),
        }
        for value, index in zip(top_values.tolist(), top_indices.tolist(), strict=False)
    ]

    return {
        "schema": "resnet-forward-trace-v1",
        "image_path": image_path,
        "true_label": true_label,
        "true_name": class_names.get(true_label, str(true_label)) if true_label is not None else None,
        "predicted_label": trace.predicted_label,
        "predicted_name": class_names.get(trace.predicted_label, str(trace.predicted_label)),
        "confidence": trace.confidence,
        "input_image_data_url": _image_to_data_url(image),
        "input_rgb_14": rgb_grid,
        "activation_maps_7": activation_maps,
        "gradcam_7": _map_to_grid(trace.gradcam_map, size=(7, 7)) if trace.gradcam_map is not None else None,
        "probabilities": probabilities,
        "stats": [
            {
                "name": stats.name,
                "shape": list(stats.shape),
                "min": stats.minimum,
                "max": stats.maximum,
                "mean": stats.mean,
                "std": stats.std,
            }
            for stats in trace.stats
        ],
    }


def save_trace_json(
    image: torch.Tensor,
    trace: PredictionTrace,
    class_names: dict[int, str],
    output_path: str | Path,
    true_label: int | None = None,
    image_path: str | None = None,
    top_k: int = 10,
) -> None:
    """Save a compact real-forward payload that can be loaded by the HTML simulator."""
    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = trace_to_browser_payload(
        image=image,
        trace=trace,
        class_names=class_names,
        true_label=true_label,
        image_path=image_path,
        top_k=top_k,
    )
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    LOGGER.info("Saved browser forward trace JSON: %s", output_path)
