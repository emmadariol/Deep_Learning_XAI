"""Run Phase 9 explanation robustness and sanity checks."""

from __future__ import annotations

import argparse
import csv
import logging
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.data import build_dataloaders, infer_class_map_path
from src.explainability_audit import (
    occlusion_sensitivity,
    randomized_copy,
    rank_correlation,
    saliency_pair_metrics,
    save_audit_grid,
    smoothgrad_saliency,
    topk_iou,
    write_csv,
)
from src.model import build_resnet50_classifier, get_device
from src.utils import set_seed, setup_logging
from src.xai import input_gradient_saliency

LOGGER = logging.getLogger("run_phase9_explainability_audit")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Audit saliency explanations with SmoothGrad, Occlusion Sensitivity "
            "and randomized-model sanity checks."
        )
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "data" / "AWA2_subset_background20" / "awa2_manifest_subset.csv",
    )
    parser.add_argument("--class-map", type=Path, default=None)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "checkpoints" / "best_resnet50_awa2.pt",
    )
    parser.add_argument("--num-examples", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--smoothgrad-samples", type=int, default=12)
    parser.add_argument("--smoothgrad-noise-std", type=float, default=0.12)
    parser.add_argument("--occlusion-patch-size", type=int, default=32)
    parser.add_argument("--occlusion-stride", type=int, default=16)
    parser.add_argument("--occlusion-batch-size", type=int, default=32)
    parser.add_argument("--top-fraction", type=float, default=0.2)
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "reports" / "phase9_explainability_audit.csv",
    )
    parser.add_argument(
        "--figure-output",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "figures" / "phase9_explainability_audit.png",
    )
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


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    checkpoint_path = checkpoint_path.expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    LOGGER.info("loaded checkpoint: %s", checkpoint_path)


def collect_correct_examples(
    model: torch.nn.Module,
    dataloader,
    device: torch.device,
    num_examples: int,
) -> tuple[torch.Tensor, torch.Tensor, list[str], list[str], list[float], list[str]]:
    """Collect correctly classified test examples for explanation auditing."""
    images_out: list[torch.Tensor] = []
    labels_out: list[torch.Tensor] = []
    names_out: list[str] = []
    pred_names_out: list[str] = []
    confidences_out: list[float] = []
    paths_out: list[str] = []
    model.eval()

    with torch.no_grad():
        for batch in dataloader:
            images = batch[0].to(device, non_blocking=True)
            labels = torch.as_tensor(batch[1], dtype=torch.long, device=device)
            names = list(batch[2])
            paths = list(batch[3])
            logits = model(images)
            probabilities = torch.softmax(logits, dim=1)
            confidences, predictions = probabilities.max(dim=1)
            for index in range(images.size(0)):
                if int(predictions[index].item()) != int(labels[index].item()):
                    continue
                images_out.append(images[index].detach().cpu())
                labels_out.append(labels[index].detach().cpu())
                names_out.append(names[index])
                pred_names_out.append(names[index])
                confidences_out.append(float(confidences[index].item()))
                paths_out.append(paths[index])
                if len(images_out) >= num_examples:
                    return (
                        torch.stack(images_out, dim=0),
                        torch.stack(labels_out, dim=0).long(),
                        names_out,
                        pred_names_out,
                        confidences_out,
                        paths_out,
                    )

    if not images_out:
        raise RuntimeError("No correctly classified examples found.")
    return (
        torch.stack(images_out, dim=0),
        torch.stack(labels_out, dim=0).long(),
        names_out,
        pred_names_out,
        confidences_out,
        paths_out,
    )


def per_example_rows(
    image_paths: list[str],
    true_names: list[str],
    predicted_names: list[str],
    confidences: list[float],
    vanilla_maps: torch.Tensor,
    smoothgrad_maps: torch.Tensor,
    occlusion_maps: torch.Tensor,
    randomized_maps: torch.Tensor,
    top_fraction: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for index in range(vanilla_maps.size(0)):
        rows.append(
            {
                "index": index,
                "image_path": image_paths[index] if index < len(image_paths) else "",
                "true_class": true_names[index],
                "predicted_class": predicted_names[index],
                "confidence": f"{confidences[index]:.6f}",
                "vanilla_vs_smoothgrad_iou_top20": f"{topk_iou(vanilla_maps[index], smoothgrad_maps[index], top_fraction):.6f}",
                "vanilla_vs_smoothgrad_spearman": f"{rank_correlation(vanilla_maps[index], smoothgrad_maps[index]):.6f}",
                "vanilla_vs_occlusion_iou_top20": f"{topk_iou(vanilla_maps[index], occlusion_maps[index], top_fraction):.6f}",
                "vanilla_vs_occlusion_spearman": f"{rank_correlation(vanilla_maps[index], occlusion_maps[index]):.6f}",
                "vanilla_vs_randomized_iou_top20": f"{topk_iou(vanilla_maps[index], randomized_maps[index], top_fraction):.6f}",
                "vanilla_vs_randomized_spearman": f"{rank_correlation(vanilla_maps[index], randomized_maps[index]):.6f}",
            }
        )
    return rows


def summary_row(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = [
        key
        for key in rows[0]
        if key.endswith("_iou_top20") or key.endswith("_spearman")
    ]
    summary: dict[str, object] = {
        "index": "summary_mean",
        "image_path": "",
        "true_class": "",
        "predicted_class": "",
        "confidence": "",
    }
    for key in numeric_keys:
        summary[key] = f"{sum(float(row[key]) for row in rows) / len(rows):.6f}"
    return summary


def main() -> None:
    args = parse_args()
    setup_logging(args.log_level)
    set_seed(args.seed)

    manifest = args.manifest.expanduser().resolve()
    class_map = args.class_map.expanduser().resolve() if args.class_map else infer_class_map_path(manifest)
    checkpoint = args.checkpoint.expanduser().resolve()
    device = get_device(args.device)
    num_classes = infer_num_classes(manifest)

    loaders = build_dataloaders(
        manifest_path=manifest,
        class_map_path=class_map,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = build_resnet50_classifier(num_classes=num_classes, pretrained=False)
    load_checkpoint(model, checkpoint, device)
    model.to(device)
    model.eval()

    images, targets, true_names, predicted_names, confidences, image_paths = collect_correct_examples(
        model=model,
        dataloader=loaders["test"],
        device=device,
        num_examples=args.num_examples,
    )
    images = images.to(device)
    targets = targets.to(device)

    LOGGER.info("selected %d correctly classified examples", images.size(0))
    vanilla_maps = input_gradient_saliency(model, images, targets)
    smoothgrad_maps = smoothgrad_saliency(
        model,
        images,
        targets,
        n_samples=args.smoothgrad_samples,
        noise_std=args.smoothgrad_noise_std,
    )
    occlusion_maps = occlusion_sensitivity(
        model,
        images,
        targets,
        patch_size=args.occlusion_patch_size,
        stride=args.occlusion_stride,
        batch_size=args.occlusion_batch_size,
    )
    random_model = randomized_copy(model, seed=args.seed + 97).to(device)
    random_model.eval()
    randomized_maps = input_gradient_saliency(random_model, images, targets)

    aggregate = {
        **saliency_pair_metrics(vanilla_maps, smoothgrad_maps, "vanilla_vs_smoothgrad", args.top_fraction),
        **saliency_pair_metrics(vanilla_maps, occlusion_maps, "vanilla_vs_occlusion", args.top_fraction),
        **saliency_pair_metrics(vanilla_maps, randomized_maps, "vanilla_vs_randomized", args.top_fraction),
    }
    LOGGER.info("aggregate metrics: %s", aggregate)

    rows = per_example_rows(
        image_paths=image_paths,
        true_names=true_names,
        predicted_names=predicted_names,
        confidences=confidences,
        vanilla_maps=vanilla_maps,
        smoothgrad_maps=smoothgrad_maps,
        occlusion_maps=occlusion_maps,
        randomized_maps=randomized_maps,
        top_fraction=args.top_fraction,
    )
    rows.append(summary_row(rows))
    write_csv(rows, args.metrics_output)
    save_audit_grid(
        images=images,
        vanilla_maps=vanilla_maps,
        smoothgrad_maps=smoothgrad_maps,
        occlusion_maps=occlusion_maps,
        randomized_maps=randomized_maps,
        true_names=true_names,
        predicted_names=predicted_names,
        confidences=confidences,
        output_path=args.figure_output,
    )
    LOGGER.info("Phase 9 complete: metrics=%s figure=%s", args.metrics_output, args.figure_output)


if __name__ == "__main__":
    main()
