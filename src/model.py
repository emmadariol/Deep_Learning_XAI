"""Model utilities for the AwA2 classifier."""

from __future__ import annotations

import logging

import torch
from torch import nn
from torchvision.models import ResNet50_Weights, resnet50

LOGGER = logging.getLogger(__name__)


def build_resnet50_classifier(
    num_classes: int,
    pretrained: bool = True,
    trainable_modules: tuple[str, ...] = ("layer4", "fc"),
) -> nn.Module:
    """Build a ResNet50 classifier with a new classification head."""
    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    for parameter in model.parameters():
        parameter.requires_grad = False

    for module_name in trainable_modules:
        module = getattr(model, module_name, None)
        if module is None:
            raise ValueError(f"ResNet50 has no module named {module_name!r}")
        for parameter in module.parameters():
            parameter.requires_grad = True

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    LOGGER.info(
        "Built ResNet50: num_classes=%d pretrained=%s trainable=%d total=%d",
        num_classes,
        pretrained,
        trainable,
        total,
    )
    return model


def get_device(requested_device: str = "auto") -> torch.device:
    """Resolve the execution device."""
    if requested_device != "auto":
        device = torch.device(requested_device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is not available.")
        if device.type == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is not available.")
        return device

    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
