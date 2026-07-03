"""Run Grad-CAM and Integrated Gradients on correctly predicted test images."""

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
    input_gradient_maps,
    integrated_gradients_maps,
    log_tensor_stats,
    saliency_concentration,
    saliency_iou_at_percentile,
    save_multi_xai_grid,
    save_xai_grid,
    spearman_rank_correlation,
    write_xai_metrics_csv,
)

LOGGER = logging.getLogger("run_xai")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Grad-CAM and IG examples.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "OxfordPets" / "oxford_pets_manifest.csv",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_oxford_pets.pt",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "xai_examples.png",
    )
    parser.add_argument(
        "--multi-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "xai_multi_class_comparison.png",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "xai_metrics.csv",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-images", type=int, default=8)
    parser.add_argument("--max-per-class", type=int, default=1)
    parser.add_argument("--ig-steps", type=int, default=16)
    parser.add_argument("--ig-internal-batch-size", type=int, default=4)
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Train the baseline before running XAI."
        )
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    LOGGER.info("Loaded checkpoint: %s", checkpoint_path)


def collect_correct_examples(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    max_images: int,
    max_per_class: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[float], list[str], list[str]]:
    images_out: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    true_names: list[str] = []
    pred_names: list[str] = []
    confidences: list[float] = []
    image_paths_out: list[str] = []
    mask_paths_out: list[str] = []
    per_class_counts: dict[int, int] = {}

    model.eval()
    with torch.no_grad():
        for batch in loader:
            images, labels, class_names = batch[0].to(device), batch[1].to(device), list(batch[2])
            image_paths = list(batch[3])
            mask_paths = list(batch[4])
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            conf, preds = probs.max(dim=1)
            correct = preds == labels

            for idx in correct.nonzero(as_tuple=False).flatten().tolist():
                label_value = int(labels[idx].item())
                if max_per_class is not None and per_class_counts.get(label_value, 0) >= max_per_class:
                    continue
                images_out.append(images[idx].detach().cpu())
                labels_out.append(labels[idx].detach().cpu())
                true_names.append(class_names[idx])
                pred_names.append(class_names[idx])
                confidences.append(float(conf[idx].detach().cpu()))
                image_paths_out.append(image_paths[idx])
                mask_paths_out.append(mask_paths[idx])
                per_class_counts[label_value] = per_class_counts.get(label_value, 0) + 1
                if len(images_out) >= max_images:
                    stacked_images = torch.stack(images_out, dim=0).to(device)
                    stacked_labels = torch.stack(labels_out, dim=0).to(device)
                    return (
                        stacked_images,
                        stacked_labels,
                        true_names,
                        pred_names,
                        confidences,
                        image_paths_out,
                        mask_paths_out,
                    )

    if not images_out:
        raise RuntimeError("No correctly predicted examples found.")
    return (
        torch.stack(images_out, dim=0).to(device),
        torch.stack(labels_out, dim=0).to(device),
        true_names,
        pred_names,
        confidences,
        image_paths_out,
        mask_paths_out,
    )


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
    LOGGER.info("Using num_classes=%d", num_classes)

    loaders = build_dataloaders(
        manifest_path=manifest,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_resnet50_classifier(num_classes=num_classes, pretrained=False)
    load_checkpoint(model, checkpoint, device)
    model.to(device)

    images, labels, true_names, _pred_names, confidences, _image_paths, _mask_paths = collect_correct_examples(
        model=model,
        loader=loaders["test"],
        device=device,
        max_images=args.max_images,
        max_per_class=args.max_per_class,
    )
    pred_names = [class_names_by_label[int(label.item())] for label in labels]

    log_tensor_stats("xai.inputs", images)
    LOGGER.info("Selected targets: %s", [int(label.item()) for label in labels])

    gradcam = GradCAM(model, model.layer4[-1])
    try:
        gradcam_maps = gradcam(images, labels)
    finally:
        gradcam.close()

    ig_maps = integrated_gradients_maps(
        model=model,
        inputs=images,
        targets=labels,
        steps=args.ig_steps,
        internal_batch_size=args.ig_internal_batch_size,
        baseline_type="blurred",
    )
    vanilla_maps, _raw_gradients = input_gradient_maps(model, images, labels)
    ig_black_maps = integrated_gradients_maps(
        model=model,
        inputs=images,
        targets=labels,
        steps=args.ig_steps,
        internal_batch_size=args.ig_internal_batch_size,
        baseline_type="black",
    )

    save_xai_grid(
        images=images,
        gradcam_maps=gradcam_maps,
        ig_maps=ig_maps,
        class_names=true_names,
        predictions=pred_names,
        confidences=confidences,
        output_path=args.output,
    )

    save_multi_xai_grid(
        images=images,
        vanilla_maps=vanilla_maps,
        gradcam_maps=gradcam_maps,
        ig_blur_maps=ig_maps,
        ig_black_maps=ig_black_maps,
        class_names=true_names,
        predictions=pred_names,
        confidences=confidences,
        output_path=args.multi_output,
    )

    gradcam_concentration = saliency_concentration(gradcam_maps, _mask_paths)
    vanilla_concentration = saliency_concentration(vanilla_maps, _mask_paths)
    ig_blur_concentration = saliency_concentration(ig_maps, _mask_paths)
    ig_black_concentration = saliency_concentration(ig_black_maps, _mask_paths)
    ig_iou = saliency_iou_at_percentile(ig_maps, ig_black_maps, percentile=80.0).detach().cpu()
    ig_spearman = spearman_rank_correlation(ig_maps, ig_black_maps).detach().cpu()

    metric_rows: list[dict[str, object]] = []
    for idx, label in enumerate(labels.detach().cpu().tolist()):
        row: dict[str, object] = {
            "index": idx,
            "label": int(label),
            "class_name": true_names[idx],
            "prediction": pred_names[idx],
            "confidence": confidences[idx],
            "image_path": _image_paths[idx],
            "mask_path": _mask_paths[idx],
            "ig_blur_vs_black_top20_iou": float(ig_iou[idx].item()),
            "ig_blur_vs_black_spearman": float(ig_spearman[idx].item()),
        }
        for prefix, values in [
            ("vanilla", vanilla_concentration[idx]),
            ("gradcam", gradcam_concentration[idx]),
            ("ig_blur", ig_blur_concentration[idx]),
            ("ig_black", ig_black_concentration[idx]),
        ]:
            row[f"{prefix}_foreground_mass"] = values["foreground_saliency_mass"]
            row[f"{prefix}_background_mass"] = values["background_saliency_mass"]
            row[f"{prefix}_boundary_mass"] = values["boundary_saliency_mass"]
        metric_rows.append(row)
    write_xai_metrics_csv(metric_rows, args.metrics_output)


if __name__ == "__main__":
    main()
