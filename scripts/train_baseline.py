"""Train the ResNet50 Oxford-IIIT Pet baseline."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders
from src.model import build_resnet50_classifier, get_device
from src.train import train_model
from src.utils import set_seed, setup_logging

LOGGER = logging.getLogger("train_baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a ResNet50 baseline on Oxford-IIIT Pet.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "OxfordPets" / "oxford_pets_manifest.csv",
        help="CSV manifest produced by scripts/prepare_oxford_pets.py.",
    )
    parser.add_argument(
        "--checkpoint-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_oxford_pets.pt",
    )
    parser.add_argument(
        "--history-path",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "training_history_resnet50_oxford_pets.csv",
    )
    parser.add_argument("--num-classes", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--grad-clip-norm", type=float, default=None)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--no-pretrained",
        action="store_true",
        help="Disable ImageNet pretrained weights. Mostly useful for offline smoke tests.",
    )
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def infer_num_classes(manifest_path: Path) -> int:
    """Infer number of labels from the manifest."""
    labels: set[int] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.add(int(row["label"]))
    if not labels:
        raise ValueError(f"No labels found in manifest: {manifest_path}")
    expected = set(range(max(labels) + 1))
    if labels != expected:
        raise ValueError(
            f"Labels are not contiguous from 0..{max(labels)}. Found: {sorted(labels)}"
        )
    return len(labels)


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    num_classes = args.num_classes or infer_num_classes(manifest)
    device = get_device(args.device)
    LOGGER.info("Using device: %s", device)
    LOGGER.info("Using manifest: %s", manifest)
    LOGGER.info("Detected num_classes=%d", num_classes)

    dataloaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet50_classifier(
        num_classes=num_classes,
        pretrained=not args.no_pretrained,
        trainable_backbone_layers=("layer3", "layer4"),
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2,
    )

    result = train_model(
        model=model,
        dataloaders=dataloaders,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        epochs=args.epochs,
        patience=args.patience,
        checkpoint_path=args.checkpoint_path,
        history_path=args.history_path,
        grad_clip_norm=args.grad_clip_norm,
    )
    LOGGER.info(
        "Best baseline: epoch=%d val_loss=%.4f val_acc=%.4f checkpoint=%s",
        result.best_epoch,
        result.best_val_loss,
        result.best_val_acc,
        args.checkpoint_path.expanduser().resolve(),
    )


if __name__ == "__main__":
    main()

