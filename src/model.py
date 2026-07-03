"""Model construction utilities for Oxford-IIIT Pet classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelSummary:
    total_params: int
    trainable_params: int
    frozen_params: int


def build_resnet50_classifier(
    num_classes: int,
    pretrained: bool = True,
    trainable_backbone_layers: tuple[str, ...] = ("layer3", "layer4"),
) -> nn.Module:
    """Build a ResNet50 classifier with a new linear head.

    By default, only layer3, layer4, and the final classifier are trainable.
    This keeps fine-tuning fast while still adapting high-level visual features
    to Oxford-IIIT Pet breeds.
    """
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    freeze_all(model)
    unfreeze_modules(model, trainable_backbone_layers)
    unfreeze_modules(model, ("fc",))

    summary = summarize_parameters(model)
    LOGGER.info(
        "Built ResNet50 classifier: num_classes=%d pretrained=%s total_params=%d trainable=%d frozen=%d",
        num_classes,
        pretrained,
        summary.total_params,
        summary.trainable_params,
        summary.frozen_params,
    )
    log_trainable_modules(model)
    return model


def freeze_all(model: nn.Module) -> None:
    """Freeze every model parameter."""
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_modules(model: nn.Module, module_names: tuple[str, ...]) -> None:
    """Unfreeze parameters belonging to named top-level modules."""
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is None:
            raise ValueError(f"Model has no module named '{module_name}'")
        for parameter in module.parameters():
            parameter.requires_grad = True


def summarize_parameters(model: nn.Module) -> ModelSummary:
    """Return total, trainable and frozen parameter counts."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    return ModelSummary(
        total_params=total,
        trainable_params=trainable,
        frozen_params=total - trainable,
    )


def log_trainable_modules(model: nn.Module) -> None:
    """Log top-level module trainability for sanity checks."""
    for name, module in model.named_children():
        params = list(module.parameters())
        if not params:
            continue
        trainable = sum(parameter.numel() for parameter in params if parameter.requires_grad)
        total = sum(parameter.numel() for parameter in params)
        LOGGER.info(
            "Module %-8s trainable_params=%d total_params=%d trainable=%s",
            name,
            trainable,
            total,
            trainable > 0,
        )


def get_device(requested_device: str = "auto") -> torch.device:
    """Resolve the execution device."""
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested_device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    return device

