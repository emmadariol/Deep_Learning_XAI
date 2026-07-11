"""Run attribution methods on selected AwA2 images."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders
from src.model import build_resnet50_classifier, get_device
from src.utils import set_seed, setup_logging
from src.xai import (
    GradCAM,
    ScoreCAM,
    expected_gradients_maps,
    integrated_gradients_maps,
    log_tensor_stats,
    save_xai_grid,
)

LOGGER = logging.getLogger("run_xai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate AwA2 attribution examples.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2" / "awa2_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "xai_examples_awa2.png",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--max-per-class", type=int, default=1)
    parser.add_argument("--ig-steps", type=int, default=50)
    parser.add_argument("--ig-internal-batch-size", type=int, default=4)
    parser.add_argument("--expected-gradient-samples", type=int, default=24)
    parser.add_argument("--expected-gradient-baselines", type=int, default=16)
    parser.add_argument("--scorecam-max-channels", type=int, default=64)
    parser.add_argument("--scorecam-batch-size", type=int, default=16)
    parser.add_argument("--blur-radius", type=float, default=18.0)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def infer_num_classes(manifest_path: Path) -> int:
    labels: set[int] = set()
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            labels.add(int(row["label"]))
    if not labels:
        raise ValueError(f"No labels found in manifest: {manifest_path}")
    return max(labels) + 1


def load_class_names(manifest_path: Path) -> dict[int, str]:
    names: dict[int, str] = {}
    with manifest_path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            names[int(row["label"])] = row["class_name"]
    return names


def load_idx_to_class(manifest_path: Path) -> dict[int, str]:
    """Compatibility alias used by later phases."""
    return load_class_names(manifest_path)


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. "
            "Run scripts/train_baseline.py before Phase 3."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded checkpoint: %s", checkpoint_path)


def collect_correct_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names_by_label: dict[int, str] | None = None,
    max_images: int = 4,
    max_per_class: int | None = None,
    idx_to_class: dict[int, str] | None = None,
    allow_incorrect: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[float], list[str]]:
    if class_names_by_label is None:
        class_names_by_label = idx_to_class
    if class_names_by_label is None:
        raise ValueError("class_names_by_label or idx_to_class must be provided.")

    images_out: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    true_names: list[str] = []
    pred_names: list[str] = []
    confidences: list[float] = []
    image_paths_out: list[str] = []
    per_class_counts: dict[int, int] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            batch_true_names = list(batch[2])
            image_paths = list(batch[3])

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            selected = torch.ones_like(labels, dtype=torch.bool) if allow_incorrect else preds == labels

            for idx in selected.nonzero(as_tuple=False).flatten().tolist():
                label_value = int(labels[idx].item())
                if max_per_class is not None and per_class_counts.get(label_value, 0) >= max_per_class:
                    continue

                images_out.append(images[idx].detach().cpu())
                labels_out.append(labels[idx].detach().cpu())
                true_names.append(batch_true_names[idx])
                pred_names.append(class_names_by_label[int(preds[idx].item())])
                confidences.append(float(conf[idx].detach().cpu().item()))
                image_paths_out.append(image_paths[idx])
                per_class_counts[label_value] = per_class_counts.get(label_value, 0) + 1

                LOGGER.info(
                    "Selected example true=%s pred=%s confidence=%.4f correct=%s image=%s",
                    true_names[-1],
                    pred_names[-1],
                    confidences[-1],
                    pred_names[-1] == true_names[-1],
                    image_paths_out[-1],
                )

                if len(images_out) >= max_images:
                    return (
                        torch.stack(images_out, dim=0).to(device),
                        torch.stack(labels_out, dim=0).to(device),
                        true_names,
                        pred_names,
                        confidences,
                        image_paths_out,
                    )

    if not images_out:
        raise RuntimeError("No selected examples found in the test split.")

    return (
        torch.stack(images_out, dim=0).to(device),
        torch.stack(labels_out, dim=0).to(device),
        true_names,
        pred_names,
        confidences,
        image_paths_out,
    )


def collect_incorrect_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    class_names_by_label: dict[int, str] | None = None,
    max_images: int = 4,
    idx_to_class: dict[int, str] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[float], list[str]]:
    """Collect misclassified examples for error-focused XAI inspection."""
    if class_names_by_label is None:
        class_names_by_label = idx_to_class
    if class_names_by_label is None:
        raise ValueError("class_names_by_label or idx_to_class must be provided.")

    images_out: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    true_names: list[str] = []
    pred_names: list[str] = []
    confidences: list[float] = []
    image_paths_out: list[str] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device, non_blocking=True)
            labels = batch[1].to(device, non_blocking=True)
            batch_true_names = list(batch[2])
            image_paths = list(batch[3])

            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            incorrect = preds != labels

            for idx in incorrect.nonzero(as_tuple=False).flatten().tolist():
                images_out.append(images[idx].detach().cpu())
                labels_out.append(labels[idx].detach().cpu())
                true_names.append(batch_true_names[idx])
                pred_names.append(class_names_by_label[int(preds[idx].item())])
                confidences.append(float(conf[idx].detach().cpu().item()))
                image_paths_out.append(image_paths[idx])
                LOGGER.info(
                    "Selected incorrect example true=%s wrong_pred=%s confidence=%.4f image=%s",
                    true_names[-1],
                    pred_names[-1],
                    confidences[-1],
                    image_paths_out[-1],
                )
                if len(images_out) >= max_images:
                    return (
                        torch.stack(images_out, dim=0).to(device),
                        torch.stack(labels_out, dim=0).to(device),
                        true_names,
                        pred_names,
                        confidences,
                        image_paths_out,
                    )

    if not images_out:
        raise RuntimeError("No misclassified examples found in the test split.")

    return (
        torch.stack(images_out, dim=0).to(device),
        torch.stack(labels_out, dim=0).to(device),
        true_names,
        pred_names,
        confidences,
        image_paths_out,
    )


def collect_reference_images(
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_images: int = 16,
) -> torch.Tensor:
    """Collect normalized images from a loader for Expected Gradients baselines."""
    images_out: list[torch.Tensor] = []
    for batch in loader:
        images = batch[0]
        for image in images:
            images_out.append(image.detach().cpu())
            if len(images_out) >= max_images:
                reference = torch.stack(images_out, dim=0).to(device)
                log_tensor_stats("expected_gradients.reference_images", reference)
                return reference
    if not images_out:
        raise RuntimeError("Could not collect reference images for Expected Gradients.")
    reference = torch.stack(images_out, dim=0).to(device)
    log_tensor_stats("expected_gradients.reference_images", reference)
    return reference


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    num_classes = infer_num_classes(manifest)
    class_names_by_label = load_class_names(manifest)

    LOGGER.info("Using device: %s", device)
    LOGGER.info("Using manifest: %s", manifest)
    LOGGER.info("Using checkpoint: %s", checkpoint)
    LOGGER.info("Detected num_classes=%d", num_classes)

    loaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet50_classifier(num_classes=num_classes, pretrained=False).to(device)
    load_checkpoint(model, checkpoint, device)
    model.eval()

    images, labels, true_names, pred_names, confidences, _image_paths = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        class_names_by_label=class_names_by_label,
        max_images=args.max_images,
        max_per_class=args.max_per_class,
    )

    log_tensor_stats("xai.inputs", images)
    LOGGER.info("Selected target labels: %s", [int(label.item()) for label in labels])

    gradcam = GradCAM(model, model.layer4[-1])
    try:
        gradcam_maps = gradcam(images, labels)
    finally:
        gradcam.close()

    scorecam = ScoreCAM(
        model=model,
        target_layer=model.layer4[-1],
        max_channels=args.scorecam_max_channels,
        batch_size=args.scorecam_batch_size,
        blur_radius=args.blur_radius,
    )
    try:
        scorecam_maps = scorecam(images, labels)
    finally:
        scorecam.close()

    ig_maps, _ig_attributions, _ig_baseline = integrated_gradients_maps(
        model=model,
        inputs=images,
        targets=labels,
        steps=args.ig_steps,
        internal_batch_size=args.ig_internal_batch_size,
        blur_radius=args.blur_radius,
    )
    expected_baselines = collect_reference_images(
        loaders["train"],
        device=device,
        max_images=args.expected_gradient_baselines,
    )
    expected_maps, _expected_attributions, _expected_baseline_pool = expected_gradients_maps(
        model=model,
        inputs=images,
        targets=labels,
        baselines=expected_baselines,
        n_samples=args.expected_gradient_samples,
        internal_batch_size=args.ig_internal_batch_size,
        blur_radius=args.blur_radius,
    )

    save_xai_grid(
        images=images,
        gradcam_maps=gradcam_maps,
        ig_maps=ig_maps,
        scorecam_maps=scorecam_maps,
        expected_gradients_maps=expected_maps,
        true_names=true_names,
        pred_names=pred_names,
        confidences=confidences,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
