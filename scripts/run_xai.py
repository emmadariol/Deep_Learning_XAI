"""Generate explicit-gradient XAI examples from a trained AwA2 checkpoint."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders, infer_class_map_path, load_class_mapping
from src.model import build_resnet50_classifier, get_device
from src.utils import set_seed, setup_logging
from src.xai import GradCAM, input_gradient_saliency, integrated_gradients, save_xai_grid

LOGGER = logging.getLogger("run_xai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Grad-CAM and explicit gradients.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2_subset_background20" / "awa2_manifest_subset.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "xai_examples.png",
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=4)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument(
        "--allow-incorrect",
        action="store_true",
        help="Allow XAI smoke tests even when examples are misclassified.",
    )
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=str, default="INFO")
    return parser.parse_args()


def load_idx_to_class(manifest_path: Path) -> dict[int, str]:
    class_map_path = infer_class_map_path(manifest_path)
    if class_map_path is None:
        raise FileNotFoundError(f"Could not infer class map for {manifest_path}")
    class_to_idx = load_class_mapping(class_map_path)
    return {label: class_name for class_name, label in class_to_idx.items()}


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Run scripts/train_baseline.py first."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    LOGGER.info("loaded checkpoint: %s", checkpoint_path)


def collect_correct_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    idx_to_class: dict[int, str],
    max_images: int,
    allow_incorrect: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[float]]:
    images_out: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    true_names: list[str] = []
    predicted_names: list[str] = []
    confidences: list[float] = []

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images = batch[0].to(device)
            labels = batch[1].to(device)
            class_names = list(batch[2])

            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)

            for index in range(images.size(0)):
                if predictions[index] != labels[index] and not allow_incorrect:
                    continue

                predicted_label = int(predictions[index].item())
                images_out.append(images[index].detach().cpu())
                labels_out.append(labels[index].detach().cpu())
                true_names.append(class_names[index])
                predicted_names.append(idx_to_class[predicted_label])
                confidences.append(float(confidence[index].detach().cpu()))

                if len(images_out) >= max_images:
                    return (
                        torch.stack(images_out).to(device),
                        torch.stack(labels_out).to(device),
                        true_names,
                        predicted_names,
                        confidences,
                    )

    if not images_out:
        raise RuntimeError(
            "No correctly predicted test examples were found. "
            "Use --allow-incorrect only for a smoke test."
        )

    return (
        torch.stack(images_out).to(device),
        torch.stack(labels_out).to(device),
        true_names,
        predicted_names,
        confidences,
    )


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    idx_to_class = load_idx_to_class(manifest)
    num_classes = len(idx_to_class)
    LOGGER.info("manifest=%s", manifest)
    LOGGER.info("num_classes=%d device=%s", num_classes, device)

    loaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )

    model = build_resnet50_classifier(
        num_classes=num_classes,
        pretrained=False,
        trainable_modules=("layer4", "fc"),
    )
    load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    images, labels, true_names, predicted_names, confidences = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        idx_to_class=idx_to_class,
        max_images=args.max_images,
        allow_incorrect=args.allow_incorrect,
    )
    LOGGER.info("selected %d examples", images.size(0))

    input_gradient_maps = input_gradient_saliency(model, images, labels)

    gradcam = GradCAM(model, model.layer4[-1])
    try:
        gradcam_maps = gradcam(images, labels)
    finally:
        gradcam.close()

    ig_maps = integrated_gradients(
        model=model,
        inputs=images,
        targets=labels,
        steps=args.ig_steps,
    )

    save_xai_grid(
        images=images,
        input_gradient_maps=input_gradient_maps,
        gradcam_maps=gradcam_maps,
        ig_maps=ig_maps,
        true_names=true_names,
        predicted_names=predicted_names,
        confidences=confidences,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
