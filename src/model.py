"""Model construction utilities for AwA2 classification."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

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
    trainable_modules: tuple[str, ...] | None = None,
) -> nn.Module:
    """Build a ResNet50 classifier with a new linear head.

    The default fine-tuning policy freezes the early visual blocks and trains
    only layer3, layer4 and the final classifier. This is fast enough for
    experiments while still adapting high-level ImageNet features to AwA2.
    """
    if trainable_modules is not None:
        trainable_backbone_layers = tuple(
            module for module in trainable_modules if module != "fc"
        )

    weights = ResNet50_Weights.DEFAULT if pretrained else None
    model = resnet50(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)

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
    for parameter in model.parameters():
        parameter.requires_grad = False


def unfreeze_modules(model: nn.Module, module_names: tuple[str, ...]) -> None:
    for module_name in module_names:
        module = getattr(model, module_name, None)
        if module is None:
            raise ValueError(f"Model has no top-level module named '{module_name}'")
        for parameter in module.parameters():
            parameter.requires_grad = True

def set_frozen_batchnorm_eval(model: nn.Module) -> None:
    """Keep frozen BatchNorm modules in eval mode during fine-tuning.

    Calling ``model.train()`` switches every BatchNorm layer back to training
    mode, even if its affine parameters are frozen. That would still update
    running mean/variance. This helper restores eval mode only for BatchNorm
    modules whose own parameters are all frozen.
    """
    batchnorm_types = (
        nn.BatchNorm1d,
        nn.BatchNorm2d,
        nn.BatchNorm3d,
        nn.SyncBatchNorm,
    )
    for module in model.modules():
        if isinstance(module, batchnorm_types):
            params = list(module.parameters(recurse=False))
            if params and all(not parameter.requires_grad for parameter in params):
                module.eval()


def summarize_parameters(model: nn.Module) -> ModelSummary:
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return ModelSummary(total, trainable, total - trainable)


def log_trainable_modules(model: nn.Module) -> None:
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


def load_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    expected_class_mapping: dict[int, str] | None = None,
) -> dict[str, Any]:
    """Safely load a state dict and validate class semantics when available."""
    path = Path(checkpoint_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Checkpoint must be a state dictionary or dictionary bundle: {path}")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    if not isinstance(state_dict, dict):
        raise TypeError(f"Checkpoint does not contain a model state dictionary: {path}")

    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        raise TypeError(f"Checkpoint metadata must be a dictionary: {path}")
    stored_mapping = metadata.get("idx_to_class")
    if expected_class_mapping is not None and stored_mapping is not None:
        normalized_mapping = {int(index): str(name) for index, name in stored_mapping.items()}
        if normalized_mapping != expected_class_mapping:
            raise ValueError(
                "Checkpoint class mapping does not match the requested dataset. "
                f"checkpoint={normalized_mapping} dataset={expected_class_mapping}"
            )
    elif expected_class_mapping is not None:
        LOGGER.warning(
            "Legacy checkpoint has no embedded class mapping; validating only tensor shapes: %s",
            path,
        )
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded checkpoint: %s", path)
    return metadata


def get_device(requested_device: str = "auto") -> torch.device:
    if requested_device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        device = torch.device(requested_device)
    except (RuntimeError, ValueError) as error:
        raise ValueError(f"Invalid PyTorch device: {requested_device!r}") from error
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device type: {device.type!r}")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available.")
    if device.type == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is not available.")
    return device
