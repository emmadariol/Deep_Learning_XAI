"""Run explanation robustness and saliency sanity checks."""

from __future__ import annotations

import argparse
import logging
import math
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.experiments.run_xai import collect_correct_examples
from src.data import (
    build_dataloaders,
    infer_class_map_path,
    infer_num_classes,
    load_class_names,
)
from src.explainability_audit import (
    occlusion_sensitivity,
    randomized_copy,
    rank_correlation,
    saliency_pair_metrics,
    save_audit_grid,
    smoothgrad_saliency,
    topk_iou,
)
from src.model import build_resnet50_classifier, get_device, load_checkpoint
from src.metrics import top_fraction_label
from src.utils import set_seed, setup_logging, write_csv
from src.validation import (
    device_spec,
    log_level,
    nonnegative_int,
    open_unit_float,
    positive_float,
    positive_int,
)
from src.xai import input_gradient_saliency

LOGGER = logging.getLogger("run_saliency_sanity_audit")


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
    parser.add_argument("--num-examples", type=positive_int, default=4)
    parser.add_argument("--batch-size", type=positive_int, default=16)
    parser.add_argument("--num-workers", type=nonnegative_int, default=0)
    parser.add_argument("--smoothgrad-samples", type=positive_int, default=12)
    parser.add_argument("--smoothgrad-noise-std", type=positive_float, default=0.12)
    parser.add_argument("--occlusion-patch-size", type=positive_int, default=32)
    parser.add_argument("--occlusion-stride", type=positive_int, default=16)
    parser.add_argument("--occlusion-batch-size", type=positive_int, default=32)
    parser.add_argument("--top-fraction", type=open_unit_float, default=0.2)
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
    parser.add_argument("--device", type=device_spec, default="auto")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-level", type=log_level, default="INFO")
    return parser.parse_args()


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
    top_label = top_fraction_label(top_fraction)
    for index in range(vanilla_maps.size(0)):
        rows.append(
            {
                "index": index,
                "image_path": image_paths[index] if index < len(image_paths) else "",
                "true_class": true_names[index],
                "predicted_class": predicted_names[index],
                "confidence": f"{confidences[index]:.6f}",
                f"vanilla_vs_smoothgrad_iou_{top_label}": f"{topk_iou(vanilla_maps[index], smoothgrad_maps[index], top_fraction):.6f}",
                "vanilla_vs_smoothgrad_spearman": f"{rank_correlation(vanilla_maps[index], smoothgrad_maps[index]):.6f}",
                f"vanilla_vs_occlusion_iou_{top_label}": f"{topk_iou(vanilla_maps[index], occlusion_maps[index], top_fraction):.6f}",
                "vanilla_vs_occlusion_spearman": f"{rank_correlation(vanilla_maps[index], occlusion_maps[index]):.6f}",
                f"vanilla_vs_randomized_iou_{top_label}": f"{topk_iou(vanilla_maps[index], randomized_maps[index], top_fraction):.6f}",
                "vanilla_vs_randomized_spearman": f"{rank_correlation(vanilla_maps[index], randomized_maps[index]):.6f}",
            }
        )
    return rows


def summary_row(rows: list[dict[str, object]]) -> dict[str, object]:
    numeric_keys = [
        key
        for key in rows[0]
        if "_iou_top" in key or key.endswith("_spearman")
    ]
    summary: dict[str, object] = {
        "index": "summary_mean",
        "image_path": "",
        "true_class": "",
        "predicted_class": "",
        "confidence": "",
    }
    for key in numeric_keys:
        values = [float(row[key]) for row in rows]
        finite_values = [value for value in values if math.isfinite(value)]
        summary[key] = (
            f"{sum(finite_values) / len(finite_values):.6f}"
            if finite_values
            else "nan"
        )
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
    class_names_by_label = load_class_names(manifest)

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
        loader=loaders["test"],
        device=device,
        class_names_by_label=class_names_by_label,
        max_images=args.num_examples,
        seed=args.seed,
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
    LOGGER.info("Saliency sanity audit complete: metrics=%s figure=%s", args.metrics_output, args.figure_output)


if __name__ == "__main__":
    main()
